import onnxruntime as ort
import numpy as np
from scipy.signal import get_window,resample_poly
from scipy.special import softmax
import random,librosa
from numpy.fft import rfft
import time
from collections import deque

def resample_(audio, new_sample_rate):
    signal, sample_rate = audio
    if sample_rate == new_sample_rate:
        return (signal, sample_rate)

    gcd = np.gcd(sample_rate, new_sample_rate)
    up = new_sample_rate // gcd
    down = sample_rate // gcd

    new_signal = resample_poly(signal, up, down, axis=-1)
    return (new_signal, new_sample_rate)

def rechannel(audio, new_channel):
    signal, sample_rate = audio
    if len(signal.shape)==1:
        new_signal=signal[np.newaxis,:]
        return (new_signal, sample_rate)
    if signal.shape[0] == new_channel:
        return audio
    if new_channel == 1:
        new_signal = signal[:1, :]
    else:
        new_signal = np.concatenate([signal, signal])
    return (new_signal, sample_rate)




def pad_trunc(audio, max_ms):
    signal, sample_rate = audio
    num_rows, sig_len = signal.shape
    max_len = sample_rate // 1000 * max_ms
    if sig_len > max_len:
        start = random.randint(0, sig_len - max_len)
        signal = signal[:, start:start + max_len]
    elif sig_len < max_len:
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len
        pad_begin = np.zeros((num_rows, pad_begin_len), dtype=np.float32)
        pad_end = np.zeros((num_rows, pad_end_len), dtype=np.float32)
        signal = np.concatenate((pad_begin, signal, pad_end), 1)
    return (signal, sample_rate)


class Generate_mel_spectrogram_np:
    def __init__(self,sr, n_fft, n_mels, f_min, f_max, norm, hop_length=None,
                 win_length=None,
        window_fn="hann",
        power=2.0,
        top_db=80,
        center=True,
        pad_mode="reflect",
        normalized=False,  # False / "window" / "frame_length"
                  ):

        self.sr=sr
        self.n_fft=n_fft
        self.n_mels=n_mels
        self.f_min=f_min
        self.f_max = f_max if f_max is not None else self.sr / 2
        self.norm=norm
        self.win_length=n_fft if win_length is None else win_length
        self.hop_length= self.win_length // 2 if hop_length is None else hop_length
        self.fbanks = self.mel_filterbank()
        # 🔹 Window function
        self.win = get_window(window_fn, self.win_length, fftbins=True).astype(np.float32)
        # 🔹 Reflect padding for center alignment
        self.pad_len = self.n_fft // 2 if center else 0
        self.power=power
        self.top_db=top_db
        self.pad_mode=pad_mode
        self.normalized=normalized


    def mel_filterbank(self):
        if self.f_max is None:
            self.f_max = self.sr / 2

        mel_min = self.hz_to_mel(self.f_min)
        mel_max = self.hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self.mel_to_hz(mel_points)

        bins = np.floor((self.n_fft + 1) * hz_points / self.sr).astype(int)
        fbanks = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for i in range(1, self.n_mels + 1):
            left = bins[i - 1]
            center = bins[i]
            right = bins[i + 1]

            if center == left:
                center += 1
            if right == center:
                right += 1

            for j in range(left, center):
                fbanks[i - 1, j] = (j - left) / (center - left)
            for j in range(center, right):
                fbanks[i - 1, j] = (right - j) / (right - center)

        if self.norm == "slaney":
            fbanks /= np.maximum(fbanks.sum(axis=1, keepdims=True), 1e-10)

        return fbanks

    @staticmethod
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    @staticmethod
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def mel_spectrogram_np(self,signal):
        # (signal_len,)
        signal = signal.astype(np.float32)

        # left and right padding: (signal_len+2*pad_len,)
        if self.pad_len > 0:
            signal = np.pad(signal, (self.pad_len, self.pad_len), mode=self.pad_mode)

        if self.normalized == "window" or self.normalized is True:
            self.win /= np.sqrt(np.sum(self.win ** 2))

        # 🔹 Framing (get each small window) and STFT
        num_frames = 1 + (len(signal) - self.n_fft) // self.hop_length
        frames = np.stack([
            signal[i * self.hop_length: i * self.hop_length + self.win_length] * self.win
            for i in range(num_frames)
        ])
        if self.normalized == "frame_length":
            frames /= np.sqrt(self.win_length)
        stft_result = np.abs(rfft(frames, n=self.n_fft)) ** self.power  # shape: (T, freq_bins)

        # 🔹 Apply Mel filter bank
        mel_spec = np.dot(stft_result, self.fbanks.T)  # shape: (T, n_mels)

        # 🔹 Convert to log scale (dB)
        mel_spec_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
        mel_spec_db = np.clip(mel_spec_db, a_min=mel_spec_db.max() - self.top_db, a_max=None)

        mel_spec_db = mel_spec_db.T  # shape: (n_mels, time_frames)
        mel_spec_db = mel_spec_db[np.newaxis, np.newaxis, ...]  # shape: (1, 1, n_mels, time)

        return mel_spec_db.astype(np.float32)

mel_generator=Generate_mel_spectrogram_np(sr=16000, n_fft=1024, n_mels=64, f_min=0.0, f_max=None, norm=None,center=False)




def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def mel_filterbank(sr, n_fft, n_mels, f_min=0.0, f_max=None,norm=None):
    if f_max is None:
        f_max = sr / 2

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbanks = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, n_mels + 1):
        left = bins[i - 1]
        center = bins[i]
        right = bins[i + 1]

        if center == left:
            center += 1
        if right == center:
            right += 1

        for j in range(left, center):
            fbanks[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbanks[i - 1, j] = (right - j) / (right - center)

    if norm == "slaney":
        fbanks /= np.maximum(fbanks.sum(axis=1, keepdims=True), 1e-10)

    return fbanks


def mel_spectrogram_np(
    signal,
    sr=16000,
    n_fft=1024,
    hop_length=None,
    win_length=None,
    n_mels=64,
    window_fn="hann",
    power=2.0,
    f_min=0.0,
    f_max=None,
    top_db=80,
    center=True,
    pad_mode="reflect",
    normalized=False,  # False / "window" / "frame_length"
    norm=None,
):
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    # (signal_len,)
    signal = signal.astype(np.float32)

    # 🔹 Reflect padding for center alignment
    pad_len = n_fft // 2 if center else 0
    # left and right padding: (signal_len+2*pad_len,)
    if pad_len > 0:
        signal = np.pad(signal, (pad_len, pad_len), mode=pad_mode)

    # 🔹 Window function
    win = get_window(window_fn, win_length, fftbins=True).astype(np.float32)
    if normalized == "window" or normalized is True:
        win /= np.sqrt(np.sum(win ** 2))

    # 🔹 Framing (get each small window) and STFT
    num_frames = 1 + (len(signal) - n_fft) // hop_length
    frames = np.stack([
        signal[i * hop_length : i * hop_length + win_length] * win
        for i in range(num_frames)
    ])

    if normalized == "frame_length":
        frames /= np.sqrt(win_length)

    stft_result = np.abs(rfft(frames, n=n_fft)) ** power  # shape: (T, freq_bins)

    # 🔹 Apply Mel filter bank
    fbanks = mel_filterbank(sr, n_fft, n_mels, f_min, f_max,  norm=norm)
    mel_spec = np.dot(stft_result, fbanks.T)  # shape: (T, n_mels)

    # 🔹 Convert to log scale (dB)
    mel_spec_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    mel_spec_db = np.clip(mel_spec_db, a_min=mel_spec_db.max() - top_db, a_max=None)

    mel_spec_db = mel_spec_db.T  # shape: (n_mels, time_frames)
    mel_spec_db = mel_spec_db[np.newaxis, np.newaxis, ...]  # shape: (1, 1, n_mels, time)

    return mel_spec_db.astype(np.float32)



def preprocess_single_audio1(audio, target_sample_rate, duration_ms, target_channel=1):
    audio = rechannel(audio, target_channel)
    audio = resample_(audio,target_sample_rate)
    audio = pad_trunc(audio,duration_ms)

    mel_spectrogram = mel_generator.mel_spectrogram_np(audio[0][0])
    #print(mel_spectrogram.shape)
    signal = audio[0]
    signal=signal[np.newaxis,:] # add batch
    return signal,mel_spectrogram



def single_predict(model_save_path,audio, hidden=None,target_sample_rate = 16000,duration_ms = 1000 ):
    # preprocess the audio signal
    t1=time.time()
    signal, mel_spectrogram = preprocess_single_audio1(audio, target_sample_rate, duration_ms)


    t2= time.time()
    #print('preprocess:',t2-t1)
    ort_session = ort.InferenceSession(model_save_path)
    if hidden is None:
        hidden_np=np.zeros((2,1,128), dtype=np.float32)
    signal,mel_spectrogram=np.float32(signal),np.float32(mel_spectrogram)
    outputs = ort_session.run(None, {
        'waveform': signal,
        'logmel': mel_spectrogram,
        'hidden': hidden_np
    })
    t3=time.time()
    #print('inference:',t3-t2)
    output,hidden=outputs[0],outputs[1]
    pred_class = np.argmax(output,1)
    output = softmax(output, axis=1)
    pred_score = np.max(output, axis=1)
    return pred_class[0],pred_score[0], hidden

def predict_whole_sequence_slide_vote(audio_path,model_save_path,chunk_ms=1000,slide_ms=250,using_hidden=False):

    signal, sr = librosa.load(audio_path)
    chunk_size = int(sr * chunk_ms / 1000)
    slide_size = int(sr * slide_ms / 1000)
    signal =signal[np.newaxis,:]
    total_len = signal.shape[1]
    # key: (start_time:end_time), values: a record array: total score for 5 category,count
    preds = dict()
    hidden = None
    for index,start in enumerate(range(0, total_len, slide_size)):
        end = start + chunk_size
        chunk = signal[:,start:end]
        #if last chunk not suffices for chunk size, padding
        if chunk.shape[1] < chunk_size:
            pad_len = chunk_size - chunk.shape[1]
            pad = np.zeros((signal.shape[0], pad_len))
            chunk = np.concatenate((chunk, pad), axis=1)
        audio_tuple = (chunk, sr)
        if using_hidden:
            pred_class, pred_score, hidden = single_predict(model_save_path, audio_tuple,hidden)
        else:

            pred_class, pred_score, _, = single_predict(model_save_path, audio_tuple, hidden)
        start_time= index* (slide_ms/ 1000)
        if int(start_time) == 10:
            break

        group1_endtime = start_time + 1*slide_ms/1000
        group2_endtime = start_time + 2*slide_ms / 1000
        group3_endtime = start_time + 3*slide_ms / 1000
        group4_endtime = start_time + 4*slide_ms / 1000
        # key exists
        try:
            preds[f'{start_time}:{group1_endtime}'][-1]+=1
            preds[f'{start_time}:{group1_endtime}'][int(pred_class)]+=pred_score
            #preds[f'{start_time}:{group1_endtime}']=mel_spectrogram
        # key not exists, create it
        except:
            record_array=np.zeros((6,))
            record_array[-1]=1
            record_array[int(pred_class)]=pred_score
            preds[f'{start_time}:{group1_endtime}'] = record_array


        try:
            preds[f'{group1_endtime}:{group2_endtime}'][-1]+=1
            preds[f'{group1_endtime}:{group2_endtime}'][int(pred_class)]+=pred_score
            #preds[f'{start_time}:{group1_endtime}'] = mel_spectrogram
        except:
            record_array2 = np.zeros((6,))
            record_array2[-1] = 1
            record_array2[int(pred_class)] = pred_score
            preds[f'{group1_endtime}:{group2_endtime}'] = record_array2

        try:
            preds[f'{group2_endtime}:{group3_endtime}'][-1]+=1
            preds[f'{group2_endtime}:{group3_endtime}'][int(pred_class)]+=pred_score
            #preds[f'{start_time}:{group1_endtime}'] = mel_spectrogram
        except:
            record_array3 = np.zeros((6,))
            record_array3[-1] = 1
            record_array3[int(pred_class)] = pred_score
            preds[f'{group2_endtime}:{group3_endtime}'] = record_array3

        try:
            preds[f'{group3_endtime}:{group4_endtime}'][-1]+=1
            preds[f'{group3_endtime}:{group4_endtime}'][int(pred_class)]+=pred_score
            #preds[f'{start_time}:{group1_endtime}'] = mel_spectrogram
        except:
            record_array4 = np.zeros((6,))
            record_array4[-1] = 1
            record_array4[int(pred_class)] = pred_score
            preds[f'{group3_endtime}:{group4_endtime}'] = record_array4
    return preds



class ContainerCache:
    def __init__(self, max_len=10):
        self.cache = deque(maxlen=max_len)

    def append(self, mel):
        """插入一個 mel 頻譜圖： (1, 1, n_mels, num_frames)
       """
        self.cache.append(mel)

    def __len__(self):
        return len(self.cache)

    def stack(self):
        """將目前所有 mel 頻譜圖疊成一個  array：(1, 1, n_mels, num_frames*2)  """
        return np.concatenate(self.cache, axis=-1)

    def __getitem__(self, slc):
        """
        slc: slice object or int
        Return a sliced tensor along time axis.
        """
        full = self.stack()
        return full[slc]



def predict_whole_sequence_slide_vote2(audio_path,model_save_path,chunk_ms=1000,slide_ms=250):
    signal, sr = librosa.load(audio_path)
    chunk_size = int(sr * chunk_ms / 1000)

    signal = signal[np.newaxis, :]
    total_len = signal.shape[1]
    #keep 2 second melspectrogram (1, 1, n_mels, num_frames*2)
    #padded_signal_len = original_len + 2 * pad_len
    # num_frames = 1 + (padded_signal_len - n_fft) // hop_length
    mel_spectrogram_cache=ContainerCache(max_len=2)
    signal_cache =ContainerCache(max_len=2)
    ort_session = ort.InferenceSession(model_save_path)


    # key: (start_time:end_time), values: a record array: total score for 5 category,count
    preds = dict()
    for index, start in enumerate(range(0, total_len, chunk_size)):
        end = start + chunk_size
        # 1 second signal
        chunk = signal[:, start:end]
        # if last chunk not suffices for chunk size, padding
        if chunk.shape[1] < chunk_size:
            pad_len = chunk_size - chunk.shape[1]
            pad = np.zeros((signal.shape[0], pad_len))
            chunk = np.concatenate((chunk, pad), axis=1)
        audio = (chunk, sr)
        audio = rechannel(audio, 1)
        audio = resample_(audio, 16000)
        audio = pad_trunc(audio, 1000)
        mel_spectrogram = mel_generator.mel_spectrogram_np(audio[0][0])

        mel_len_per = int(mel_spectrogram.shape[-1]/(chunk_ms / slide_ms))
        #sig_len_per =   int(len(audio[0][0]) /(chunk_ms / slide_ms))

        #print(mel_spectrogram.shape)

        mel_spectrogram_cache.append(mel_spectrogram)
        signal_cache.append(audio[0][0].reshape((1,1,1,16000)))

        #get the prediction and refresh the cache stack
        if len(mel_spectrogram_cache)==2 and len(signal_cache)==2:
            # return 5 sliding window results
            start_time = (index - 1) * (chunk_ms / 1000)
            if int(start_time)==10:
                break
            if int(start_time) == 0:
                start_ind=0
            else:
                start_ind=1

            for sub_ind in range(start_ind,5):
                group1_endtime =sub_ind*250/1000+ start_time + 1 * slide_ms / 1000
                group2_endtime =sub_ind*250/1000+ start_time + 2 * slide_ms / 1000
                group3_endtime =sub_ind*250/1000+ start_time + 3 * slide_ms / 1000
                group4_endtime =sub_ind*250/1000+ start_time + 4 * slide_ms / 1000

                signal_start_time= sub_ind*250+start_time*1000
                signal_end_time = signal_start_time+ 1000
                signal_start = int(signal_start_time/1000  * audio[1])
                signal_end =   int(signal_end_time/1000 * audio[1])
                #signal_start= int(sub_ind* sig_len_per)
                #signal_end =  int(signal_start+sig_len_per*(chunk_ms / slide_ms))
                signal_ =  signal[:,signal_start:signal_end]


                mel_start= int(sub_ind* mel_len_per)
                mel_end = int(mel_start+ mel_len_per*(chunk_ms / slide_ms))

                mel_spectrogram_= mel_spectrogram_cache[:,:,:,mel_start:mel_end]

                hidden_np = np.zeros((2, 1, 128), dtype=np.float32)
                signal_, mel_spectrogram_ = np.float32(signal_), np.float32(mel_spectrogram_)
                outputs = ort_session.run(None, {
                    'waveform': signal_.reshape((1,1,signal_.shape[-1])),
                    'logmel': mel_spectrogram_,
                    'hidden': hidden_np
                })
                output, hidden = outputs[0], outputs[1]
                pred_class = np.argmax(output, 1)
                output = softmax(output, axis=1)
                pred_score = np.max(output, axis=1)


                # key exists
                try:
                    #preds[f'{start_time}:{group1_endtime}'] = mel_spectrogram
                    preds[f'{sub_ind*250/1000+start_time}:{group1_endtime}'][-1] +=1
                    preds[f'{sub_ind*250/1000+start_time}:{group1_endtime}'][int(pred_class)] += pred_score
                # key not exists, create it
                except:
                    record_array = np.zeros((6,))
                    record_array[-1] = 1
                    record_array[int(pred_class)] = pred_score
                    preds[f'{sub_ind*250/1000+start_time}:{group1_endtime}'] = record_array

                try:
                    #preds[f'{start_time}:{group1_endtime}'] = mel_spectrogram
                    preds[f'{group1_endtime}:{group2_endtime}'][-1] += 1
                    preds[f'{group1_endtime}:{group2_endtime}'][int(pred_class)] += pred_score
                except:
                    record_array2 = np.zeros((6,))
                    record_array2[-1] = 1
                    record_array2[int(pred_class)] = pred_score
                    preds[f'{group1_endtime}:{group2_endtime}'] = record_array2

                try:
                    #preds[f'{start_time}:{group1_endtime}'] = mel_spectrogram
                    preds[f'{group2_endtime}:{group3_endtime}'][-1] +=1
                    preds[f'{group2_endtime}:{group3_endtime}'][int(pred_class)] += pred_score
                except:
                    record_array3 = np.zeros((6,))
                    record_array3[-1] = 1
                    record_array3[int(pred_class)] = pred_score
                    preds[f'{group2_endtime}:{group3_endtime}'] = record_array3

                try:
                    #preds[f'{start_time}:{group1_endtime}'] = mel_spectrogram
                    preds[f'{group3_endtime}:{group4_endtime}'][-1] += 1
                    preds[f'{group3_endtime}:{group4_endtime}'][int(pred_class)] += pred_score
                except:
                    record_array4 = np.zeros((6,))
                    record_array4[-1] = 1
                    record_array4[int(pred_class)] = pred_score
                    preds[f'{group3_endtime}:{group4_endtime}'] = record_array4

    return preds





def test_mel_generator(audio_path,chunk_ms=1000,slide_ms=250):
    mel_list1,mel_list2=[],[]

    signal, sr = librosa.load(audio_path)
    chunk_size = int(sr * chunk_ms / 1000)
    slide_size = int(sr * slide_ms / 1000)
    signal = signal[np.newaxis, :]
    total_len = signal.shape[1]

    for index, start in enumerate(range(0, total_len, slide_size)):
        end = start + chunk_size
        chunk = signal[:, start:end]
        audio_tuple = (chunk, sr)
        audio_tuple = rechannel(audio_tuple, 1)
        audio_tuple = resample_(audio_tuple, 16000)
        audio_tuple = pad_trunc(audio_tuple, 1000)
        mel_spectrogram_ = mel_generator.mel_spectrogram_np(audio_tuple[0][0])
        mel_list1.append(mel_spectrogram_)
        if len(mel_list1)==5:
            break


    mel_spectrogram_cache = ContainerCache(max_len=2)
    for index, start in enumerate(range(0, total_len, chunk_size)):
        end = start + chunk_size
        chunk = signal[:, start:end]
        audio = (chunk, sr)
        audio = rechannel(audio, 1)
        audio = resample_(audio, 16000)
        audio = pad_trunc(audio, 1000)
        mel_spectrogram = mel_generator.mel_spectrogram_np(audio[0][0])
        mel_spectrogram_cache.append(mel_spectrogram)
        mel_len_per = round(mel_spectrogram.shape[-1] / (chunk_ms / slide_ms))
        if len(mel_spectrogram_cache)==2:
            for sub_ind in range(0, 5):
                mel_start = int(sub_ind * mel_len_per)
                mel_end = int(mel_start + mel_len_per * (chunk_ms / slide_ms))
                mel_spectrogram_ = mel_spectrogram_cache[:, :, :, mel_start:mel_end]
                mel_list2.append(mel_spectrogram_)
        if index==1:
            break

    return mel_list1,mel_list2





















if __name__ =='__main__':
    audio_path = '726_16k.wav'
    model_save_path="waveform_logmel_cnn_gru.onnx"

    # cache=MelSpectrogramCache(max_len=2)
    # mel_1=np.ones((1,1,64,32))
    # mel_2=np.zeros((1,1,64,32))
    # cache.append(mel_1)
    # cache.append(mel_2)
    # print(f"Current length: {len(cache)}")

    #preds = predict_whole_sequence_slide_vote(audio_path, model_save_path)
    #preds2 = predict_whole_sequence_slide_vote2(audio_path, model_save_path)
    mel_list1, mel_list2 =test_mel_generator(audio_path)
    for i,j in zip(mel_list1,mel_list2):
        print(np.allclose(i,j,atol=1e-5))


    # class_names_list = ['A', 'C', 'D', 'N', 'S']
    # with open('726_cache.txt', 'w') as f:
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


