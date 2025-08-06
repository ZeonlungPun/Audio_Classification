import numpy as np
from WaveformLogmelDatasets import AudioDataset
import torch,torchaudio
from WaveformLogmel import WaveformLogmelClassifier,WaveformLogmelCNNGRUClassifier
from onnx_inference import preprocess_single_audio1
def preprocess_single_audio(audio, target_sample_rate, duration_ms, target_channel=1):
    #signal, sample_rate = torchaudio.load(audio)
    #audio = (signal, sample_rate)
    # predict the audio with specific length (300ms)
    # 2. 調整為指定通道
    audio = AudioDataset.rechannel(audio, target_channel)

    # 3. 重取樣
    audio = AudioDataset.resample(audio, target_sample_rate)

    # 4. 補齊或裁剪為固定時長（毫秒）
    audio = AudioDataset.pad_trunc(audio, duration_ms)


    # 5. 轉為 Mel Spectrogram
    mel_spectrogram = AudioDataset.spectro_gram(audio)

    # 6. 加上 batch 維度與 channel 維度 (1, 1, n_mels, time)
    if mel_spectrogram.ndim == 2:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)  # 加 channel
    mel_spectrogram = mel_spectrogram.unsqueeze(0)  # 加 batch
    signal = audio[0]
    signal=signal.unsqueeze(0)
    return signal,mel_spectrogram

def single_predict(model,save_path,audio, hidden=None,target_sample_rate = 16000,duration_ms = 1000 ):
    # 預處理音訊
    signal, mel_spectrogram = preprocess_single_audio(audio, target_sample_rate, duration_ms)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    # 推論
    model.eval()
    with torch.no_grad():
        output, hidden = model(signal, mel_spectrogram,hidden)
        pred_class = torch.argmax(output, dim=1).item()
        output =torch.softmax(output,dim=1)
        pred_score = torch.max(output,dim=1)[0].item()
    return pred_class,pred_score, hidden

def predict_whole_sequence(audio_path,model,save_path,chunk_ms=1000,using_hidden=False):
    # 載入音訊：得到 (signal, sample_rate)
    signal, sr = torchaudio.load(audio_path)
    # 計算每段的點數 (e.g. 0.3s × sr)
    chunk_size = int(sr * chunk_ms / 1000)

    total_len = signal.shape[1]  # 總樣本數
    preds = []

    hidden =None
    for start in range(0, total_len, chunk_size):

        end = start + chunk_size
        chunk = signal[:, start:end]

        # 若最後一段長度不足 chunk_size，補零
        if chunk.shape[1] < chunk_size:
            pad_len = chunk_size - chunk.shape[1]
            pad = torch.zeros((signal.shape[0], pad_len))
            chunk = torch.cat((chunk, pad), dim=1)
        audio_tuple=(chunk,sr)
        if using_hidden:
            pred_class,pred_score, hidden=single_predict(model,save_path,audio_tuple,hidden)
        else:
            pred_class, pred_score, _ = single_predict(model, save_path, audio_tuple, hidden)
        preds.append((pred_class,pred_score))
    return preds

def predict_whole_sequence_slide(audio_path,model,save_path,chunk_ms=1000,slide_ms=250,using_hidden=False):
    # 載入音訊：得到 (signal, sample_rate)
    signal, sr = torchaudio.load(audio_path)
    # 計算每段的點數 (e.g. 0.3s × sr)
    chunk_size = int(sr * chunk_ms / 1000)
    slide_size = int(sr * slide_ms / 1000)
    total_len = signal.shape[1]
    preds=[]
    hidden = None
    for start in range(0, total_len, slide_size):
        end = start + chunk_size
        chunk = signal[:, start:end]
        if chunk.shape[1] < chunk_size:
            pad_len = chunk_size - chunk.shape[1]
            pad = torch.zeros((signal.shape[0], pad_len))
            chunk = torch.cat((chunk, pad), dim=1)
        audio_tuple = (chunk, sr)
        if using_hidden:
            pred_class, pred_score, hidden = single_predict(model, save_path, audio_tuple,hidden)
        else:
            pred_class, pred_score, _ = single_predict(model, save_path, audio_tuple, hidden)
        preds.append((pred_class, pred_score))
    return preds


def predict_whole_sequence_slide_vote(audio_path,model,save_path,chunk_ms=1000,slide_ms=250,using_hidden=False):

    signal, sr = torchaudio.load(audio_path)
    chunk_size = int(sr * chunk_ms / 1000)
    slide_size = int(sr * slide_ms / 1000)
    total_len = signal.shape[1]
    # key: (start_time:end_time), values: a record array: total score for 5 category,count
    preds = dict()
    hidden = None
    for index,start in enumerate(range(0, total_len, slide_size)):
        end = start + chunk_size
        chunk = signal[:, start:end]
        if chunk.shape[1] < chunk_size:
            pad_len = chunk_size - chunk.shape[1]
            pad = torch.zeros((signal.shape[0], pad_len))
            chunk = torch.cat((chunk, pad), dim=1)
        audio_tuple = (chunk, sr)
        if using_hidden:
            pred_class, pred_score, hidden = single_predict(model, save_path, audio_tuple,hidden)
        else:
            pred_class, pred_score, _ = single_predict(model, save_path, audio_tuple, hidden)
        start_time= index* (slide_ms/ 1000)
        group1_endtime = start_time + 1*slide_ms/1000
        group2_endtime = start_time + 2*slide_ms / 1000
        group3_endtime = start_time + 3*slide_ms / 1000
        group4_endtime = start_time + 4*slide_ms / 1000
        # key exists
        try:
            preds[f'{start_time}:{group1_endtime}'][-1]+=1
            preds[f'{start_time}:{group1_endtime}'][int(pred_class)]+=pred_score
        # key not exists, create it
        except:
            record_array=np.zeros((6,))
            record_array[-1]=1
            record_array[int(pred_class)]=pred_score
            preds[f'{start_time}:{group1_endtime}'] = record_array

        try:
            preds[f'{group1_endtime}:{group2_endtime}'][-1]+=1
            preds[f'{group1_endtime}:{group2_endtime}'][int(pred_class)]+=pred_score
        except:
            record_array2 = np.zeros((6,))
            record_array2[-1] = 1
            record_array2[int(pred_class)] = pred_score
            preds[f'{group1_endtime}:{group2_endtime}'] = record_array2

        try:
            preds[f'{group2_endtime}:{group3_endtime}'][-1]+=1
            preds[f'{group2_endtime}:{group3_endtime}'][int(pred_class)]+=pred_score
        except:
            record_array3 = np.zeros((6,))
            record_array3[-1] = 1
            record_array3[int(pred_class)] = pred_score
            preds[f'{group2_endtime}:{group3_endtime}'] = record_array3

        try:
            preds[f'{group3_endtime}:{group4_endtime}'][-1]+=1
            preds[f'{group3_endtime}:{group4_endtime}'][int(pred_class)]+=pred_score
        except:
            record_array4 = np.zeros((6,))
            record_array4[-1] = 1
            record_array4[int(pred_class)] = pred_score
            preds[f'{group3_endtime}:{group4_endtime}'] = record_array4
    return preds






def compare_torch_onnx(model, onnx_model_path, input_audio_path):
    import onnxruntime as ort
    from scipy.special import softmax
    import librosa
    # 預處理（PyTorch）
    signal1, mel1 = preprocess_single_audio(torchaudio.load(input_audio_path), target_sample_rate=16000, duration_ms=1000)
    hidden1 = torch.zeros(2, 1, 128)

    # 預處理（NumPy for ONNX）
    signal2, mel2 = preprocess_single_audio1( librosa.load(audio_path), target_sample_rate=16000, duration_ms=1000)
    signal2_np, mel2_np = np.float32(signal2), np.float32(mel2)
    hidden2_np = np.zeros((2, 1, 128), dtype=np.float32)

    # 比對 signal 差異
    signal_diff = np.mean(np.abs(signal1.numpy() - signal2_np))
    print("📊 Signal mean abs diff:", signal_diff)

    # 比對 logmel 差異
    mel_diff = np.mean(np.abs(mel1.numpy() - mel2_np))
    print("📊 LogMel mean abs diff:", mel_diff)

    # 推論 PyTorch
    model.eval()
    with torch.no_grad():
        torch_out, _ = model(signal1, mel1, hidden1)
        torch_softmax = torch.softmax(torch_out, dim=1).numpy()

    # 推論 ONNX
    ort_sess = ort.InferenceSession(onnx_model_path)
    ort_out = ort_sess.run(None, {
        "waveform": signal2_np,
        "logmel": mel2_np,
        "hidden": hidden2_np
    })[0]
    ort_softmax = softmax(ort_out, axis=1)

    # 比對 Softmax 輸出差異
    softmax_diff = np.mean(np.abs(torch_softmax - ort_softmax))
    print("✅ Softmax mean abs diff:", softmax_diff)
if __name__ =='__main__':
    save_path = './clean_dataset/crnn-2stream.pth'
    class_names_list = ['read-aloud', 'clap', 'discuss', 'noise', 'single']
    #model = WaveformLogmelClassifier(num_classes=5)
    model =WaveformLogmelCNNGRUClassifier(num_classes=5)
    #audio_path='/home/zonekey/project/test/source5/test4.wav'
    audio_path = 'teacher_ac1.mp3'
    preds=predict_whole_sequence(audio_path, model, save_path)
    #preds = predict_whole_sequence_slide_vote(audio_path, model, save_path)
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    #onnx_path="waveform_logmel_cnn_gru.onnx"
    #compare_torch_onnx(model, onnx_path, audio_path)


    # with open('726.txt', 'w') as f:
    #     for k in preds.keys():
    #
    #         record_array=preds[k]
    #         time_array=k.split(':')
    #         t1,t2=float(time_array[0]),float(time_array[1])
    #         pred_class=int(np.argmax(record_array[0:-1]))
    #         score_ =  np.max(record_array[0:-1])
    #         if score_< 1:
    #             score=score_
    #         elif score_>=1 and score_ <=2:
    #             score=score_/2
    #         elif score_>2 and score_ <=3:
    #             score=score_/3
    #         else:
    #             score=score_/4
    #         f.write(f"{t1:.3f}\t{t2:.3f}\t{class_names_list[pred_class]} {score:.2f}\n")




    with open('window_4.txt','w') as f:
        for i in range(len(preds)):
            start_time=i* (1000/1000)

            end_time=start_time+(1000/1000)
            score=preds[i][1]
            category=class_names_list[preds[i][0]]

            if score< 0.4 and i!=0:
                score = preds[i-1][1]
                category = class_names_list[preds[i-1][0]]
                f.write(f"{start_time:.3f}\t{end_time:.3f}\t{category} \n \n")
            elif score< 0.4 and i==0:
                score = preds[i +1][1]
                category = class_names_list[preds[i + 1][0]]
                f.write(f"{start_time:.3f}\t{end_time:.3f}\t{category} \n \n")
            else:
                f.write(f"{start_time:.3f}\t{end_time:.3f}\t{category} \n \n")




    # audio_dir = "/home/zonekey/project/audio_classification/clean_dataset/val_manual/noise"
    # error_dict={}
    # audio_path_list=os.listdir(audio_dir)
    # for audio_name in audio_path_list:
    #     audio_path=os.path.join(audio_dir,audio_name)
    #     pred_class=single_predict(model,save_path,audio_path)
    #     if pred_class != 3:
    #         error_dict[audio_name]=pred_class
    # print('finish')


