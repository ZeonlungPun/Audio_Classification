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






int main(int argc, char** argv) {
    const char *fname = "/home/zonekey/project/audio_classification/teacher_ac1.mp3";
    std::vector<int16_t> samples = load_pcm_using_ffmpeg(fname);
    printf("got %zu samples, with %.03f seconds\n", samples.size(), samples.size() / 16000.0);    


    const int16_t *ptr_sample = samples.data();

    size_t head = 0, tail = samples.size();

    int sr = 16000;
    double freq = 440.0;
    double duration = 1.0;

    cv::Mat signal = generate_sine(sr, freq, duration);

    Generate_mel_spectrogram mel_generator(16000,512,64,false,0.0);

    cv::Mat w=mel_generator.forward(signal);

    
    
    while (head < tail) {
        // PCM pcm(ptr_sample + head, ptr_sample + head + 16000);
        PCM pcm(1, 16000, CV_16S, (void*)(ptr_sample + head));
        //printf("got %zu samples with %.03f seconds \n",pcm.size(),pcm.size()/16000);
        head += 16000;
        
    }

    return 0;
}