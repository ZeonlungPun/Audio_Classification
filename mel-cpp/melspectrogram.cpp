#include <vector>
#include <stdio.h>
#include <iostream>
#include <math.h>
extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswresample/swresample.h>
}
#include <opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <string>
typedef cv::Mat PCM;

cv::Mat generate_sine(int sr, float freq, float duration) {
    int N = static_cast<int>(sr * duration);
    cv::Mat signal = cv::Mat::zeros(1, N, CV_32F);

    for (int n = 0; n < N; n++) {
        float t = static_cast<float>(n) / sr;
        signal.at<float>(0, n) = std::sin(2.0 * M_PI * freq * t);
    }

    return signal;  // shape (1, N)
}


std::vector<int16_t> load_pcm_using_ffmpeg(const char *fname, float duration=100) {
    // 使用 ffmpeg，从 fname 中加载所有声音数据，并且转换为 f32, mono, 16k 的数据
    std::vector<int16_t> samples;
    //100 秒 × 16000 samples/秒 = 1,600,000 個 sample
    //預先分配記憶體容量
    samples.reserve(duration * 16000);  

    AVFormatContext *fc = 0;  //檔案管理員」→ 知道整個檔案有幾條音訊/影片流。
    AVCodecContext *cc = 0;  //專門用來解碼一條音訊流的引擎。
    
    //打開一個媒體檔案，並建立對應的 AVFormatContext。
    int rc = avformat_open_input(&fc, fname, 0, 0);
    if (rc < 0) {
        std::cerr << "cannot open file '" << fname << "'" << std::endl;
    }
    //分析檔案內的所有 stream，收集它們的詳細資訊。
    rc = avformat_find_stream_info(fc, 0);
    if (rc < 0) {
        std::cout<<"cannot find stream info \n"<<std::endl;
    }

    int audio_stream_index = -1;
    //在檔案裡找到音訊流，並建立對應的解碼器
    for (int i = 0; i < fc->nb_streams; ++i) {
        AVStream *st = fc->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            // 找到對應的解碼器
            const AVCodec *codec = avcodec_find_decoder(fc->streams[i]->codecpar->codec_id);
            if (!codec) {
                std::cout<<"codec not found\n"<<std::endl;
            }
            //建立解碼器上下文，cc 會保存解碼需要的所有狀態（取樣率、聲道、buffer…）
            cc = avcodec_alloc_context3(codec);
            //  複製參數到解碼器
            rc = avcodec_parameters_to_context(cc, st->codecpar);
            if (rc < 0) {
                std::cout<<"cannot copy codec parameters\n"<<std::endl;
            }
            //打開解碼器
            rc = avcodec_open2(cc, codec, 0);
            if (rc < 0) {
                std::cout<<"cannot open codec\n"<<std::endl;
            }
            //記錄音訊流索引
            audio_stream_index = i;
            break;
        }
    }

    if (audio_stream_index < 0) {
        std::cout<<"cannot find audio stream\n"<<std::endl;
    }
//壓縮數據包，用來存放從檔案讀出來的編碼數據（例如 MP3、AAC frame）。
    AVPacket *pkt = av_packet_alloc();
// 解壓後的音訊幀，存放解碼後的 PCM 數據。
// 用來存放從解碼器得到的「原始幀」，可能是多聲道、不同取樣率、不同格式。   
    AVFrame *src_pcm = av_frame_alloc();
//用來存放經過重採樣後的「統一格式的幀」（單聲道、16kHz、S16P）。
    AVFrame *dst_pcm = av_frame_alloc();
//. 設定目標音訊格式:指定採樣格式 → 有符號 16-bit、平面格式 (planar)。
    dst_pcm->format = AV_SAMPLE_FMT_S16P;
//設定聲道布局
#if (LIBAVCODEC_VERSION_MAJOR <= 58)
    dst_pcm->channel_layout = AV_CH_LAYOUT_MONO;
#else
    dst_pcm->ch_layout = AV_CHANNEL_LAYOUT_MONO;
#endif // 

//設定取樣率
    dst_pcm->sample_rate = 16000;
//設定每個 frame 的大小
    dst_pcm->nb_samples = 16000; 
//分配 buffer
    rc = av_frame_get_buffer(dst_pcm, 0);
    if (rc < 0) {
        std::cout<<"cannot alloc buffer\n"<<std::endl;
    }
    //SwrContext → FFmpeg 的 音訊重採樣器（libswresample 提供）
    struct SwrContext *swr = 0;
    bool quit = false;

    while (!quit) {
        //→ 從檔案讀一個「壓縮的 packet」（例如 AAC/MP3 frame）。
        rc = av_read_frame(fc, pkt);
        if (rc < 0) {
            break;
        }
        // 過濾非音訊流
        if (pkt->stream_index != audio_stream_index) {
            av_packet_unref(pkt);
            continue;
        }
        //送 packet 進解碼器
        rc = avcodec_send_packet(cc, pkt);
        if (rc < 0) {
            std::cout<<"cannot send packet\n"<<std::endl;
        }

        //從解碼器取出 frame
        while (1) {
        //avcodec_receive_frame 嘗試取出一個解碼後的音訊幀，存到 src_pcm。
            rc = avcodec_receive_frame(cc, src_pcm);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) {
                break;
            } else if (rc < 0) {
                std::cout<<"cannot receive frame\n"<<std::endl;
            }

            if (src_pcm->channel_layout == 0) {
                if (src_pcm->channels == 1) {
                    src_pcm->channel_layout = AV_CH_LAYOUT_MONO;
                }
                else if (src_pcm->channels == 2) {
                    src_pcm->channel_layout = AV_CH_LAYOUT_STEREO;
                }
                else {
                    throw std::runtime_error("NOT impl!");
                }
            }

            AVFrame *pcm = 0;
        //判斷是否需要重採樣
            if (src_pcm->format != dst_pcm->format || 
#if (LIBAVCODEC_VERSION_MAJOR <= 58)
                src_pcm->channel_layout != dst_pcm->channel_layout ||
#else
                src_pcm->ch_layout.nb_channels != dst_pcm->ch_layout.nb_channels ||
#endif // 
                src_pcm->sample_rate != dst_pcm->sample_rate
            ) {
                // 需要 resample
                if (!swr) {
#if (LIBAVCODEC_VERSION_MAJOR <= 58)
                    swr = swr_alloc_set_opts(swr, 
                        dst_pcm->channel_layout, (AVSampleFormat)dst_pcm->format, dst_pcm->sample_rate,
                        src_pcm->channel_layout, (AVSampleFormat)src_pcm->format, src_pcm->sample_rate,
                        0, nullptr);

#else
                    swr_alloc_set_opts2(&swr,
                        &dst_pcm->ch_layout, (AVSampleFormat)dst_pcm->format, dst_pcm->sample_rate,
                        &src_pcm->ch_layout, (AVSampleFormat)src_pcm->format, src_pcm->sample_rate,
                        0, 0
                    );
#endif // 
                    int rc;
                    if ((rc = swr_init(swr)) < 0) {
                        std::cout<<"cannot init swr "<<std::endl;
                        char buf[1024];
                        av_strerror(rc, buf, sizeof(buf));
                        printf("swr_init fail:%d(%s)\n", rc, buf);
                    }
                }

                rc = swr_convert_frame(swr, dst_pcm, src_pcm);
                pcm = dst_pcm;
            }
            else {
                // 不需要 resample
                pcm = src_pcm;
            }

            // 将 pcm 中的数据复制到 samples 中
            int nb_samples = pcm->nb_samples;
            int16_t *data = (int16_t *)pcm->extended_data[0];
            for (int i = 0; i < nb_samples; ++i) {
                samples.push_back(data[i]);
            }

            if (samples.size() / 16000 >= duration) {
                quit = true;
                break;
            }
        }
    }
//資源釋放
    av_frame_free(&src_pcm);
    av_frame_free(&dst_pcm);
    av_packet_free(&pkt);
    avcodec_free_context(&cc);
    avformat_close_input(&fc);
    if (swr) {
        swr_free(&swr);
    }

    return samples;
}



class Generate_mel_spectrogram
{ 
public:
    enum class PadMode { CONSTANT, REFLECT, REPLICATE };
    int sample_rate;
    int n_fft;
    int n_mels;
    bool norm;
    bool center;
    int f_min;
    int f_max;
    int win_length;
    int hop_length;
    int pad_len;
    int power;
    int top_db;
    
    std::string pad_mode;//"constant","reflect","replicate"
    cv::Mat fbanks;
    cv::Mat win;
    


    Generate_mel_spectrogram(
        int sample_rate, int n_fft, int n_mels, bool norm,
        int f_min, int f_max = -1, int win_length = -1, int hop_length = -1,
        int pad_len = 0, std::string pad_mode = "reflect",
        int power = 2, int top_db = 80, bool center = true)
        : sample_rate(sample_rate), n_fft(n_fft), n_mels(n_mels), norm(norm),
          f_min(f_min),f_max(f_max),win_length(win_length),hop_length(hop_length), 
          power(power), top_db(top_db), center(center),pad_len(pad_len),
          pad_mode(pad_mode)
    {
        // Python: self.win_length = n_fft if win_length is None else win_length
        this->win_length = (win_length == -1) ? n_fft : win_length;

        // self.f_max = sample_rate/2 if f_max is None else f_max
        this->f_max = (f_max == -1) ? sample_rate / 2 : f_max;

        // self.hop_length = win_length//2 if hop_length is None else hop_length
        this->hop_length = (hop_length == -1) ? this->win_length / 2 : hop_length;

        // self.pad_len = n_fft//2 if center else 0
        this->pad_len = (center) ? n_fft / 2 : 0;

        //初始化 fbanks 和 window
        this->fbanks = mel_filterbank();
        this->win = get_window();
    }

    PadMode set_pad_mode(const std::string& mode) {
        PadMode pad_mode_enum;
        if (mode == "constant") pad_mode_enum = PadMode::CONSTANT;
        else if (mode == "reflect") pad_mode_enum = PadMode::REFLECT;
        else if (mode == "replicate") pad_mode_enum = PadMode::REPLICATE;
        else throw std::invalid_argument("Unsupported pad_mode: " + mode);
        return pad_mode_enum;
    }

    cv::Mat mel_filterbank()
    {   
        float mel_min = this->hz_to_mel((float)this->f_min);
        float mel_max = this->hz_to_mel((float)this->f_max);

        std::vector<int> bins(this->n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++) {
            float mel = mel_min + (mel_max - mel_min) * i / (this->n_mels + 1);
            float hz  = mel_to_hz(mel);
            bins[i] = static_cast<int>((this->n_fft + 1) * hz / this->sample_rate);
        }
        
        cv::Mat fbanks = cv::Mat::zeros(this->n_mels, this->n_fft / 2 + 1, CV_32F);

        for (int i=1;i<this->n_mels+1;i++)
        {
            int left=bins[i-1];
            int center_=bins[i];
            int right=bins[i+1];

            if (center_==left)
            {
                center_+=1;
            }
            if (right==center_)
            {
                right+=1;
            }

            for (int j=left;j<center_;j++)
            {
                fbanks.at<float>(i-1,j)=(static_cast<float>(j - left)) / (center_ - left);
            }

            for (int j=center_;j<right;j++)
            {
                fbanks.at<float>(i-1,j)= (static_cast<float>(right - j)) / (right - center_);  
            }

        }

    
        
        return fbanks;
    }

    cv::Mat get_window()
    {
        //get Hann window
        int M = this->win_length;
        bool sym = false; 
        std::vector<double> a = {0.5, 0.5}; 

        // extend
        int M_ext = M;
        bool needs_trunc = false;
        if (!sym) {
            M_ext = M + 1;
            needs_trunc = true;
        }

        // fac = linspace(-pi, pi, M_ext)
        std::vector<float> fac;
        fac.reserve(M_ext);
        for (int i=0; i < M_ext; i++)
        {
            fac.push_back(-M_PI + (2.0 * M_PI * i) / (M_ext - 1));
        }

        // w = sum a[k] * cos(k*fac)
        cv::Mat w = cv::Mat::zeros(1, M_ext, CV_32F);
        for (int k=0; k < (int)a.size(); k++)
        {
            for (int j=0; j < M_ext; j++)
            {
                w.at<float>(0,j) += a[k] * cos(k * fac[j]);
            }
        }

        if (needs_trunc) {
            return w(cv::Range::all(), cv::Range(0, M));
        }
        return w;
    }
    
    float hz_to_mel(float hz)
    {
        return 2595*log10(1+hz/700);
    }

    float mel_to_hz(float mel)
    {
        return 700*(std::pow(10,mel/2595)-1);
    }  

    cv::Mat frame_signal(cv::Mat signal)
    {
        int num_frames = 1 + (signal.cols - this->win_length) / this->hop_length;
        cv::Mat frames = cv::Mat::zeros(num_frames, this->win_length, CV_32F);

        for (int i = 0; i < num_frames; i++)
        {
            cv::Mat multiple_result;
            
            cv::Mat segment = signal(cv::Range::all(),
                                    cv::Range(i * this->hop_length,
                                            i * this->hop_length + this->win_length));
            cv::multiply(segment, this->win, multiple_result, 1.0, CV_32F);
            multiple_result.copyTo(frames.row(i));
        }

        return frames;
    }

    cv::Mat forward(cv::Mat signal)
    {
        //left and right padding: (signal_len+2*pad_len,)
        if(this->pad_len>0)
        {
            PadMode PadModenum=this->set_pad_mode(this->pad_mode);
            switch (PadModenum)
            {
            case PadMode::CONSTANT:
                cv::copyMakeBorder(signal, signal,
                0, 0,              // 上下不補
                pad_len, pad_len,  // 左右補 pad_len
                cv::BORDER_CONSTANT, 
                cv::Scalar(0));    // constant 值 = 0
                break;
            case  PadMode::REFLECT:
                cv::copyMakeBorder(signal,signal,
                0, 0,
                pad_len, pad_len,
                cv::BORDER_REFLECT_101);
                break;  
            case PadMode::REPLICATE:
                cv::copyMakeBorder(signal, signal,
                0, 0,
                pad_len, pad_len,
                cv::BORDER_REPLICATE);
                break;
            default:
                throw std::invalid_argument("Unsupported pad_mode");
                break;
            }
        }
        //🔹 Framing (get each small window)
        cv::Mat frames = this->frame_signal(signal);

        //Compute the one-dimensional discrete Fourier Transform for real input.
        int num_frames = frames.rows;
        cv::Mat spec(num_frames, n_fft/2 + 1, CV_32FC2); 

        for (int i=0;i<num_frames;i++)
        {
            cv::Mat fft_result;
            cv::dft(frames.row(i),fft_result,cv::DFT_COMPLEX_OUTPUT);
            // 只取前 n_fft//2+1 頻率
            cv::Mat half = fft_result.colRange(0, n_fft/2 + 1).clone();
            half.copyTo(spec.row(i));
        }
        // compute the magnitude： sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
        std::vector<cv::Mat> channels(2);
        cv::split(spec, channels);
        cv::Mat real_part = channels[0];  
        cv::Mat imag_part = channels[1];

        cv::Mat real_result,imag_result,manitude_result;
        
        cv::multiply(real_part,real_part,real_result,1.0, CV_32F);
        cv::multiply(imag_part,imag_part,imag_result,1.0, CV_32F);
        cv::add(real_result,imag_result,manitude_result);

        //🔹 Apply Mel filter bank
        cv::Mat mel_spec = manitude_result* this->fbanks.t();
        //🔹 Convert to log scale (dB)
        cv::max(mel_spec, 1e-10, mel_spec);
        cv::Mat mel_spec_db;
        cv::log(mel_spec,mel_spec_db);
        mel_spec_db=10*mel_spec_db/std::log(10.0);
        double maxVal;
        cv::Point maxLoc;
        cv::minMaxLoc(mel_spec_db, NULL,&maxVal,NULL,&maxLoc);
        cv::max(mel_spec_db,maxVal-this->top_db,mel_spec_db);

        mel_spec_db=mel_spec_db.t();

        
        return mel_spec_db;

    }
};

std::vector<std::string> labels = {"read-aloud", "clap", "discuss", "noise", "single"};

std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float sum_exp = 0.0f;

    //std::cout << "--- Softmax 函數內部偵錯 ---" << std::endl;

    // 計算所有 logits 的指數和
    for (size_t i = 0; i < logits.size(); ++i) {
        float logit = logits[i];
        // 檢查 logit 是否為 NaN 或 Inf
        if (std::isnan(logit)) {
            std::cerr << "偵錯：Softmax 輸入 logit[" << i << "] 是 NaN。" << std::endl;
            // 如果輸入是 NaN，則直接返回 NaN 向量
            std::fill(probabilities.begin(), probabilities.end(), std::numeric_limits<float>::quiet_NaN());
            return probabilities;
        }
        if (std::isinf(logit)) {
            std::cerr << "偵錯：Softmax 輸入 logit[" << i << "] 是 Inf。" << std::endl;
            // 對於 Inf 輸入，結果通常也是 Inf 或 NaN，這裡直接返回 NaN 向量
            std::fill(probabilities.begin(), probabilities.end(), std::numeric_limits<float>::quiet_NaN());
            return probabilities;
        }

        float exp_logit = std::exp(logit);
        //std::cout << "  logit[" << i << "]: " << logit << ", exp(logit): " << exp_logit << std::endl;
        sum_exp += exp_logit;
    }

    //std::cout << "  所有 exp(logit) 的總和 (sum_exp): " << sum_exp << std::endl;

    // 檢查 sum_exp 是否為 0 或 NaN
    if (sum_exp == 0.0f || std::isnan(sum_exp) || std::isinf(sum_exp)) {
        std::cerr << "偵錯：Softmax 計算中 sum_exp 為 " << sum_exp << "。這會導致除以零或 NaN/Inf 結果。" << std::endl;
        // 如果 sum_exp 有問題，則返回 NaN 向量
        std::fill(probabilities.begin(), probabilities.end(), std::numeric_limits<float>::quiet_NaN());
        return probabilities;
    }

    // 計算每個類別的機率
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i]) / sum_exp;
    }
    //std::cout << "--- Softmax 函數內部偵錯結束 ---" << std::endl;
    return probabilities;
}

std::pair<int,float> parse_model_output(const std::vector<Ort::Value>& ort_outputs) {
    if (ort_outputs.empty()) {
        std::cerr << "錯誤：模型沒有輸出。" << std::endl;
        return std::make_pair(0,0);
    }

    const Ort::Value& output_tensor = ort_outputs[0];
    std::vector<int64_t> output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

    // 這裡檢查形狀是否符合 [1, 5]
    if (output_shape.size() != 2 || output_shape[0] != 1 || output_shape[1] != 5) {
        std::cerr << "錯誤：輸出張量形狀不符合預期的 1x5 分類任務。" << std::endl;
        std::cerr << "輸出形狀：[";
        for (size_t i = 0; i < output_shape.size(); ++i) {
            std::cerr << output_shape[i] << (i == output_shape.size() - 1 ? "" : "x");
        }
        std::cerr << "]" << std::endl;
        return std::make_pair(0,0);
    }

    const float* raw_output_data = output_tensor.GetTensorData<float>();

    std::vector<float> class_scores(5);
    for (int i = 0; i < 5; ++i) {
        class_scores[i] = raw_output_data[i];
    }

    // --- 關鍵偵錯點：打印模型的原始輸出 ---
    // std::cout << "\n--- 模型原始輸出 (Logits/Scores) 偵錯 ---" << std::endl;
    // for (int i = 0; i < class_scores.size(); ++i) {
    //     std::cout << "  類別 " << i << " 的原始分數: " << class_scores[i] << std::endl;
    // }
    // std::cout << "--- 模型原始輸出偵錯結束 ---\n" << std::endl;
    // --- 結束關鍵偵錯點 ---

    std::vector<float> probabilities = softmax(class_scores);

    float max_probability = 0.0f;
    int predicted_class_index = -1;

    // 檢查機率向量是否包含 NaN
    bool has_nan_prob = false;
    for (float prob : probabilities) {
        if (std::isnan(prob)) {
            has_nan_prob = true;
            break;
        }
    }

    if (has_nan_prob) {
        std::cerr << "錯誤：機率計算結果包含 NaN。無法確定預測類別。" << std::endl;
        predicted_class_index = -1; // 表示無法確定
    } else {
        for (int i = 0; i < probabilities.size(); ++i) {
            if (probabilities[i] > max_probability) {
                max_probability = probabilities[i];
                predicted_class_index = i;
            }
        }
    }


    //std::cout << "每個類別的機率：" << std::endl;
    for (int i = 0; i < probabilities.size(); ++i) {
        // 如果機率是 NaN，則輸出 "NaN"
        if (std::isnan(probabilities[i])) {
            std::cout << "  類別 " << i << ": NaN%" << std::endl;
        }
        // } else {
        //     std::cout << "  類別 " << i << ": " << probabilities[i] * 100.0f << "%" << std::endl;
        // }
    }

    //std::cout << "---" << std::endl;
    // if (predicted_class_index != -1) {
    //     std::cout << "預測的類別索引是: " << predicted_class_index << std::endl;
    //     std::cout << "預測的機率是: " << max_probability * 100.0f << "%" << std::endl;
    // } else {
    //     std::cout << "無法確定預測類別，因為機率包含 NaN。" << std::endl;
    // }
    // std::cout << "---" << std::endl;
    std::pair<int,float> result=std::make_pair(predicted_class_index,max_probability);
    return result;

}

std::pair<int,float> main_inference_process(const std::string& onnx_path_name, PCM pcm, Generate_mel_spectrogram& mel_generator)
{
    // 初始化 ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "sound_classification-onnx");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session session(env, onnx_path_name.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // 取得輸入輸出節點名稱
    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::vector<std::string> input_node_names_str;
    std::vector<std::string> output_node_names_str;
    input_node_names_str.reserve(numInputNodes);
    output_node_names_str.reserve(numOutputNodes);

    int input_w = 0;
    int input_h = 0;
    for (size_t i = 0; i < numInputNodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_node_names_str.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        input_w = input_dims[3];
        input_h = input_dims[2];
        // std::cout << "Input " << i << " format: NxCxHxW = " 
        //           << input_dims[0] << "x" << input_dims[1] << "x" 
        //           << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    int class_num = output_dims[1];

    // std::cout << "Output format: BxClass_num = " 
    //           << output_dims[0] << "x" << output_dims[1] << std::endl;

    for (size_t i = 0; i < numOutputNodes; i++) {
        auto out_name = session.GetOutputNameAllocated(i, allocator);
        output_node_names_str.push_back(out_name.get());
    }
    
   

    // ===== 生成 Mel Spectrogram =====
    cv::Mat mel_spectrogram = mel_generator.forward(pcm);

    int mel_T = mel_spectrogram.rows;
    int mel_F = mel_spectrogram.cols;
    

    //create input tensor
    const std::array<const char*, 2> inputNames = { input_node_names_str[0].c_str(),input_node_names_str[1].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names_str[0].c_str() };
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    int tpixels1 = pcm.cols;
    std::array<int64_t, 4> input_shape_info1{1,1, 1,tpixels1};

    int tpixels2 = mel_T* mel_F;
    std::array<int64_t, 4> input_shape_info2{1,1,mel_T,mel_F};

    Ort::Value input_tensor_1 = Ort::Value::CreateTensor<float>(allocator_info, pcm.ptr<float>(), tpixels1, input_shape_info1.data(), input_shape_info1.size());
    Ort::Value input_tensor_2 = Ort::Value::CreateTensor<float>(allocator_info, mel_spectrogram.ptr<float>(), tpixels2, input_shape_info2.data(), input_shape_info2.size());
    
    // Create a vector to hold all input Ort::Value objects
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(input_tensor_1)); // Use std::move for efficiency if input_tensor_1 won't be used again
    input_tensors.push_back(std::move(input_tensor_2)); // Use std::move for efficiency if input_tensor_2 won't be used again

    // Pass the data pointer of the vector and its size to session->Run
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session.Run(Ort::RunOptions{ nullptr },
                                inputNames.data(),       // Array of input names
                                input_tensors.data(),    // Pointer to the array of input tensors
                                input_tensors.size(),    // Number of input tensors
                                outNames.data(),         // Array of output names
                                outNames.size());        // Number of output tensors
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::pair<int,float> pred_result = parse_model_output(ort_outputs);





    return pred_result;
}



int main(int argc, char** argv) {
    const char *fname = "/home/zonekey/project/audio_classification/726_16k.wav";
    std::vector<int16_t> samples = load_pcm_using_ffmpeg(fname,40*60);
    printf("got %zu samples, with %.03f seconds\n", samples.size(), samples.size() / 16000.0);    


    const int16_t *ptr_sample = samples.data();

    size_t head = 0, tail = samples.size();

    int sr = 16000;
    double freq = 440.0;
    double duration = 1.0;

    cv::Mat signal = generate_sine(sr, freq, duration);

    Generate_mel_spectrogram mel_generator(16000,512,64,false,0.0);
    std::string onnx_path_name="/home/zonekey/project/audio_classification/waveform_logmel_cnn.onnx";
    std::ofstream outputFile("726_c.txt");
    
    int time=0;
    while (head < tail) {
        // PCM pcm(ptr_sample + head, ptr_sample + head + 16000);
        PCM pcm(1, 16000, CV_16S, (void*)(ptr_sample + head));
        //printf("got %zu samples with %.03f seconds \n",pcm.size(),pcm.size()/16000);
       
        cv::Mat pcm_float_mat(1, pcm.cols, CV_32F);
        float* pcm_float_ptr = pcm_float_mat.ptr<float>();
        for (int i = 0; i < pcm.cols; ++i) {
            int16_t sample_value = pcm.at<int16_t>(0,head + i);
            pcm_float_ptr[i] = static_cast<float>(sample_value) / 32768.0f;
        }
        
        std::pair<int,float> pred_result =main_inference_process(onnx_path_name,pcm_float_mat,mel_generator);
        if (outputFile.is_open()) {
            int pred_class = pred_result.first;
            float score = pred_result.second;
            // 計算浮點數時間，並將其格式化
            float start_time_float = static_cast<float>(time);
            float end_time_float = static_cast<float>(time + 1);
            std::string category = labels[pred_class];

            // 使用 std::fixed 和 std::setprecision 來格式化輸出
            outputFile << std::fixed << std::setprecision(3) << start_time_float << "\t"
                       << std::fixed << std::setprecision(3) << end_time_float << "\t"
                       << category << " \n \n"; 
        }
        head += 16000;
        time+=1;
        
    }
    outputFile.close();
   

    return 0;
}