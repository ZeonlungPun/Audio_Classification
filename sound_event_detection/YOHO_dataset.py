import os,torch,random,math
import soundfile as sf
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

def construct_examples(audio_path, win_len = 2.5, hop_len = 1.0, sr = 16000.0):

    """
    :param audio_path:音訊檔案的路徑。
    :param win_len:每個音訊片段（window）的長度（秒），預設為 2.5 秒。
    :param hop_len:相鄰片段之間的間隔（秒），預設為 1 秒。
    :param sr:採樣率，預設
    :return:
    """

    #把時間（秒）轉換為樣本點數（samples），即將秒數乘以採樣率。
    win_len = int(sr*win_len)
    hop_len = int(sr*hop_len)

    a, sr = sf.read(audio_path)
    if len(a.shape)==2 and a.shape[1]==2:
        a=a[:,0]
    #補齊音訊長度
    if a.shape[0] < win_len:

       a_padded = np.zeros((win_len, ))
       a_padded[0:a.shape[0]] = a

    #否則就計算需要幾次 hop 來覆蓋整段音訊，並補零到足夠長的總長度。
    else:

        no_of_hops = math.ceil((a.shape[0] - win_len) / hop_len)
        a_padded = np.zeros((int(win_len + hop_len*no_of_hops), ))
        a_padded[0:a.shape[0]] = a
    #從音訊中滑動提取每段長度為 win_len 的片段，每次滑動 hop_len。
    a_ex = [a_padded[i - win_len : i] for i in range(win_len, a_padded.shape[0]+1, hop_len)]
    #建立對應的時間區段
    win_ranges = [((i - win_len)/sr, i/sr) for i in range(win_len, a_padded.shape[0]+1, hop_len)]

    return a_ex, win_ranges


def construct_labels(events, win_start, win_end, win_len):
    """
從標註檔中提取在某段時間範圍內的事件，並對其進行合併與整理，可用於對應音訊片段的 supervision label。
    :param events: np.array list, 每一行代表： 類別 開始時間 結束時間
    :param win_start:目前窗口的起始時間（秒）。
    :param win_end:目前窗口的結束時間（秒）。
    :param win_len:win_len: 窗口的總長度（win_end - win_start，以防錯位）。
    :return:
    [[start1, end1, class1],
     [start2, end2, class2],
        ...]
    """
    # ann: 開始時間,結束時間,類別名稱
    ann = [[float(e[1]), float(e[2]), int(e[0])] for e in events]
    curr_ann = []
#篩選出當前視窗內有交集的事件，篩選條件：只保留那些在目前 window 內有「重疊」的事件。
#並且把事件時間轉換為相對於當前窗口的時間（從 0 開始算）。
    for a in ann:
        if a[1] > win_start and a[0] <= win_end:
            curr_start = max(a[0] - win_start, 0.0)
            curr_end = min(a[1] - win_start, win_len)
            curr_ann.append([curr_start, curr_end, a[2]])
#將事件依照類別分組。
#例如：{"dog_bark": [...], "car_horn": [...]}
    class_set = set([c[2] for c in curr_ann])
    class_wise_events = {}
    for c in list(class_set):
        class_wise_events[c] = []
    for c in curr_ann:
        class_wise_events[c[2]].append(c)

# 合併重疊或間距很短的事件（最大容忍靜默時間）
    max_event_silence = 0.0
    all_events = []

    for k in list(class_wise_events.keys()):
        curr_events = class_wise_events[k]
        count = 0

        while count < len(curr_events) - 1:
            if (curr_events[count][1] >= curr_events[count + 1][0]) or (
                    curr_events[count + 1][0] - curr_events[count][1] <= max_event_silence):
                curr_events[count][1] = max(curr_events[count + 1][1], curr_events[count][1])
                del curr_events[count + 1]
            else:
                count += 1

        all_events += curr_events

    for i in range(len(all_events)):
        all_events[i][0] = round(all_events[i][0], 3)
        all_events[i][1] = round(all_events[i][1], 3)

    all_events.sort(key=lambda x: x[0])

    return all_events


def get_universal_labels(events, class_dict, ex_length=2.5, no_of_div=32):
    """
    :param events: 輸入事件標註清單，如 [start, end, class_name]
    :param class_dict: 將類別名稱對應到整數（如 "dog_bark": 0）。
    :param ex_length: 每段音訊的總長度（秒），預設為 10 秒。
    :param no_of_div: 將音訊等分為幾格（bin），預設為 32。
    :return:
    """
    # 每個 bin 時間長度（例如 10秒 / 32 = 0.3125 秒）
    win_length = ex_length / no_of_div
    #每個類別對應 3 個值（1 個 presence + 2 個 regression 值）。
    # 所以對 5 類輸出為 shape (no_of_div, 15)。
    labels = np.zeros((no_of_div, len(class_dict.keys()) * 3))
#對每個事件標註生成對應 bin 標籤
    for e in events:
        start_time = float(e[0])
        stop_time = float(e[1])
#計算事件落在哪些 bin,用整除 // 找到事件對應的起始與終止 bin。
        start_bin = int(start_time // win_length)
        stop_bin = int(stop_time // win_length)

        start_time_2 = start_time - start_bin * win_length
        stop_time_2 = stop_time - stop_bin * win_length

        n_bins = stop_bin - start_bin
#單一 bin 包含整個事件
        if n_bins == 0:
            labels[start_bin, e[2] * 3:e[2] * 3 + 3] = [1, start_time_2, stop_time_2]
#事件跨兩個 bin
        elif n_bins == 1:
            labels[start_bin, e[2] * 3:e[2] * 3 + 3] = [1, start_time_2, win_length]

            if stop_time_2 > 0.0:
                labels[stop_bin, e[2] * 3:e[2] * 3 + 3] = [1, 0.0, stop_time_2]
#事件跨多個 bin
        elif n_bins > 1:
            labels[start_bin, e[2] * 3:e[2] * 3 + 3] = [1, start_time_2, win_length]

            for i in range(1, n_bins):
                labels[start_bin + i, e[2] * 3:e[2] * 3 + 3] = [1, 0.0, win_length]

            if stop_time_2 > 0.0:
                labels[stop_bin, e[2] * 3:e[2] * 3 + 3] = [1, 0.0, stop_time_2]

    # labels[:, [1, 2, 4, 5]] /= win_length
#將 start、stop 時間正規化（歸一化到 0～1）
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if j % 3 != 0:
                labels[i][j] /= win_length
    return labels


class YOHODataset(Dataset):
    def __init__(self,audio_dir,class_list,augment=False):
        self.audio_dir=audio_dir
        self.class_list=class_list
        self.augment = augment
        self.num_classes = len(class_list)
        self.wav_dir = os.path.join(self.audio_dir,'wav')
        self.label_dir = os.path.join(self.audio_dir,'label')


    def __len__(self):
        return len(self.wav_dir)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig = aud
        top_db = 80
        spec = transforms.MelSpectrogram(16000, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def add_background_noise(signal, noise_level=0.005):
        noise = torch.randn_like(signal) * noise_level
        return signal + noise

    @staticmethod
    def random_gain(signal, min_gain=0.8, max_gain=1.2):
        gain = random.uniform(min_gain, max_gain)
        return signal * gain

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

    def __getitem__(self, index):
        wave_list = os.listdir(self.wav_dir)
        wave_file_name = wave_list[index]
        wave_full_path=os.path.join(self.wav_dir,wave_file_name)
        label_file_name = wave_file_name.replace('.wav', '.txt')
        label_full_path=os.path.join(self.label_dir,label_file_name)
        events=[]
        with open(label_full_path,'r') as f:
            all_lines=f.readlines()
        for line in all_lines:
            events.append(list(map(float, line.strip('\n').split(' '))))
        events=np.array(events)
        class_dict = {label_name: index for index, label_name in enumerate(self.class_list)}

        # in our case just 1 example and 1 label
        a_ex, win_ranges= construct_examples(audio_path=wave_full_path, win_len=2.56, hop_len=1.96)
        #  [start_time, end_time]
        w = win_ranges[0]
        waveform =a_ex[0]

        labels_t=construct_labels(events, win_start=w[0], win_end=w[1], win_len=2.56)
        labels_array = get_universal_labels(labels_t,class_dict, ex_length=2.56, no_of_div=6)
        labels_array_tensor= torch.tensor(labels_array, dtype=torch.float32)
        waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
        if self.augment and random.random() < 0.3:
            gain_db = random.uniform(-25, -5)  # dB 降低音量，更貼近真實場景
            factor = 10.0 ** (gain_db / 20.0)
            waveform_tensor = waveform_tensor * factor
        if self.augment and random.random() < 0.3:
            waveform_tensor = self.random_gain(waveform_tensor)
            waveform_tensor = self.add_background_noise(waveform_tensor)
        spec = self.spectro_gram(waveform_tensor, n_mels=64, n_fft=1024, hop_len=512)
        if self.augment and random.random() < 0.25:
            spec = self.spectro_augment(spec, max_mask_pct=0.2, n_freq_masks=2, n_time_masks=2)
        if self.augment is False:
            return spec,labels_array_tensor,label_full_path,w
        else:
        # spec: (n_mels, n_T) ,(num_divs, 3*class_num)
            return spec,labels_array_tensor

if __name__ == '__main__':
    data_dir ='/home/zonekey/project/audio_classification/sound_event_detection/synthetic_dataset/train'
    class_names_list = ['aloud', 'clap', 'discuss', 'noise', 'single']
    dataset=YOHODataset(audio_dir=data_dir,class_list=class_names_list,augment=False)
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,)

    for spec,labels_array,label_full_path,w in dataloader:
        print(spec.shape)
        print(labels_array.shape)
        print(len(label_full_path))
        print(len(w))
        print(w)
        break