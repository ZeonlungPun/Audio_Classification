import os
from WaveformLogmelDatasets import AudioDataset
import torch,torchaudio
from WaveformLogmel import LogmelCNNGRUClassifier

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
    return mel_spectrogram

def single_predict(model,save_path,audio,target_sample_rate = 16000,duration_ms = 1000 ):
    # 預處理音訊
    mel_spectrogram = preprocess_single_audio(audio, target_sample_rate, duration_ms)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    # 推論
    model.eval()
    with torch.no_grad():
        output = model( mel_spectrogram)
        pred_class = torch.argmax(output, dim=1).item()
        output =torch.softmax(output,dim=1)
        pred_score = torch.max(output,dim=1)[0].item()
    return pred_class,pred_score

def predict_whole_sequence(audio_path,model,save_path,chunk_ms=1000):
    # 載入音訊：得到 (signal, sample_rate)
    signal, sr = torchaudio.load(audio_path)
    # 計算每段的點數 (e.g. 0.3s × sr)
    chunk_size = int(sr * chunk_ms / 1000)

    total_len = signal.shape[1]  # 總樣本數
    preds = []

    for start in range(0, total_len, chunk_size):

        end = start + chunk_size
        chunk = signal[:, start:end]

        # 若最後一段長度不足 chunk_size，補零
        if chunk.shape[1] < chunk_size:
            pad_len = chunk_size - chunk.shape[1]
            pad = torch.zeros((signal.shape[0], pad_len))
            chunk = torch.cat((chunk, pad), dim=1)
        audio_tuple=(chunk,sr)
        pred_class,pred_score=single_predict(model,save_path,audio_tuple)
        preds.append((pred_class,pred_score))
    return preds

if __name__ =='__main__':
    save_path = './crnn-stream512nfft.pth'
    class_names_list = ['A', 'C', 'D', 'N', 'S']
    model =LogmelCNNGRUClassifier(num_classes=5)
    audio_path='726_16k.wav'
    preds=predict_whole_sequence(audio_path, model, save_path)

    with open('726_16k.txt','w') as f:
        for i in range(len(preds)):
            start_time=i* (1000/1000)

            end_time=start_time+(1000/1000)
            score=preds[i][1]
            category=class_names_list[preds[i][0]]

            if score< 0.65 and i!=0:
                score = preds[i-1][1]
                category = class_names_list[preds[i-1][0]]
                f.write(f"{start_time:.3f}\t{end_time:.3f}\t{category} {score:.2f}\n \n")
            elif score< 0.65 and i==0:
                score = preds[i +1][1]
                category = class_names_list[preds[i + 1][0]]
                f.write(f"{start_time:.3f}\t{end_time:.3f}\t{category} {score:.2f}\n \n")
            else:
                f.write(f"{start_time:.3f}\t{end_time:.3f}\t{category} {score:.2f}\n \n")




