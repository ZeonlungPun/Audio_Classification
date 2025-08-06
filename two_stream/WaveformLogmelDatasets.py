import os,torch,torchaudio,random
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, wave_path_list, target_sample_rate, duration, class_names_list, target_channel=1, augment=False):
        self.wave_path_list = wave_path_list
        self.duration = duration
        self.sample_rate = target_sample_rate
        self.channel = target_channel
        self.shift_pct = 0.4
        self.class_names_list = class_names_list
        self.augment=augment


    def __len__(self):
        return len(self.wave_path_list)

    @staticmethod
    def rechannel(audio, new_channel):
        signal, sample_rate = audio
        if signal.shape[0] == new_channel:
            return audio
        if new_channel == 1:
            new_signal = signal[:1, :]
        else:
            new_signal = torch.cat([signal, signal])
        return (new_signal, sample_rate)

    @staticmethod
    def resample(audio, new_sample_rate):
        signal, sample_rate = audio
        if sample_rate == new_sample_rate:
            return audio
        num_channels = signal.shape[0]
        new_signal = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(signal[:1, :])
        if num_channels > 1:
            new_signal_two = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(signal[1:, :])
            new_signal = torch.cat([new_signal, new_signal_two])
        return (new_signal, new_sample_rate)

    @staticmethod
    def pad_trunc(audio, max_ms):
        signal, sample_rate = audio
        num_rows, sig_len = signal.shape
        max_len = sample_rate // 1000 * max_ms
        if sig_len > max_len:
            #start = random.randint(0, sig_len - max_len)
            start=0
            signal = signal[:, start:start + max_len]
        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            signal = torch.cat((pad_begin, signal, pad_end), 1)
        return (signal, sample_rate)

    @staticmethod
    def time_shift(audio, shift_limit):
        sig, sr = audio
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def add_background_noise(signal, noise_level=0.005):
        noise = torch.randn_like(signal) * noise_level
        return signal + noise

    @staticmethod
    def random_gain(signal, min_gain=0.8, max_gain=1.2):
        gain = random.uniform(min_gain, max_gain)
        return signal * gain

    @staticmethod
    def random_polarity_inversion(signal):
        if random.random() < 0.5:
            return -signal
        return signal

    @staticmethod
    def time_stretch(audio, rate_range=(0.8, 1.2)):
        signal, sr = audio
        rate = random.uniform(*rate_range)
        stretched = transforms.TimeStretch()(signal.unsqueeze(0))
        return (stretched.squeeze(0), sr)

    @staticmethod
    def pitch_shift(audio, n_steps_range=(-2, 2)):
        signal, sr = audio
        n_steps = random.uniform(*n_steps_range)
        shift_transform = torchaudio.transforms.PitchShift(sr, n_steps=n_steps)
        return (shift_transform(signal), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
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
        wave_path = self.wave_path_list[index]
        class_name = wave_path.split(os.sep)[-2]
        label = self.class_names_list.index(class_name)
        signal, sample_rate = torchaudio.load(wave_path)
        audio = (signal, sample_rate)
        if label == 4 and random.random() < 0.5 and self.augment:
            signal = signal * 0.5
        audio = (signal, sample_rate)

        # 基本處理
        audio = self.resample(audio, self.sample_rate)
        audio = self.rechannel(audio, self.channel)
        audio = self.pad_trunc(audio, self.duration)
        spec = self.spectro_gram(audio, n_mels=64, n_fft=512, hop_len=None)

        signal, _ = audio
        if self.augment:
            if random.random() < 0.15:
                signal = self.random_gain(signal)
                signal = self.add_background_noise(signal)
                #signal = self.random_polarity_inversion(signal)
                audio = (signal, self.sample_rate)

                #audio = self.time_shift(audio, self.shift_pct)
                #audio = self.pitch_shift(audio, n_steps_range=(-2, 2))
                spec = self.spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return spec, audio[0], label


class AudioDataset2(Dataset):
    def __init__(self, wave_path_list, target_sample_rate, duration, class_names_list, target_channel=1, augment=False):
        self.wave_path_list = wave_path_list
        self.duration = duration
        self.sample_rate = target_sample_rate
        self.channel = target_channel
        self.shift_pct = 0.4
        self.class_names_list = class_names_list
        self.augment=augment

    def __len__(self):
        return len(self.wave_path_list)

    @staticmethod
    def rechannel(audio, new_channel):
        signal, sample_rate = audio
        if signal.shape[0] == new_channel:
            return audio
        if new_channel == 1:
            new_signal = signal[:1, :]
        else:
            new_signal = torch.cat([signal, signal])
        return (new_signal, sample_rate)

    @staticmethod
    def resample(audio, new_sample_rate):
        signal, sample_rate = audio
        if sample_rate == new_sample_rate:
            return audio
        num_channels = signal.shape[0]
        new_signal = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(signal[:1, :])
        if num_channels > 1:
            new_signal_two = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(signal[1:, :])
            new_signal = torch.cat([new_signal, new_signal_two])
        return (new_signal, new_sample_rate)

    @staticmethod
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
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            signal = torch.cat((pad_begin, signal, pad_end), 1)
        return (signal, sample_rate)

    @staticmethod
    def time_shift(audio, shift_limit):
        sig, sr = audio
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def add_background_noise(signal, noise_level=0.005):
        noise = torch.randn_like(signal) * noise_level
        return signal + noise

    @staticmethod
    def random_gain(signal, min_gain=0.8, max_gain=1.2):
        gain = random.uniform(min_gain, max_gain)
        return signal * gain

    @staticmethod
    def random_polarity_inversion(signal):
        if random.random() < 0.5:
            return -signal
        return signal

    @staticmethod
    def time_stretch(audio, rate_range=(0.8, 1.2)):
        signal, sr = audio
        rate = random.uniform(*rate_range)
        stretched = transforms.TimeStretch()(signal.unsqueeze(0))
        return (stretched.squeeze(0), sr)

    @staticmethod
    def pitch_shift(audio, n_steps_range=(-2, 2)):
        signal, sr = audio
        n_steps = random.uniform(*n_steps_range)
        shift_transform = torchaudio.transforms.PitchShift(sr, n_steps=n_steps)
        return (shift_transform(signal), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
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
        wave_path = self.wave_path_list[index]
        class_name = wave_path.split(os.sep)[-2]
        label = self.class_names_list.index(class_name)
        signal, sample_rate = torchaudio.load(wave_path)
        if self.augment and label ==4 and random.random() < 0.5:
            gain_db = random.uniform(-25, -5)  # dB 降低音量，更貼近真實場景
            factor = 10.0 ** (gain_db / 20.0)
            signal = signal * factor
        if self.augment and label != 4 and random.random() < 0.3:
            gain_db = random.uniform(-10, -3)
            factor = 10.0 ** (gain_db / 20.0)
            signal = signal * factor
        audio = (signal, sample_rate)



        # 預處理原始音訊
        audio = self.resample(audio, self.sample_rate)
        audio = self.rechannel(audio, self.channel)
        audio = self.pad_trunc(audio, self.duration)
        signal, _ = audio

        # --- Mixup on waveform ---
        if self.augment and random.random() < 0.65:
            # 隨機取另一段音訊
            mix_idx = random.randint(0, len(self.wave_path_list) - 1)
            mix_wave_path = self.wave_path_list[mix_idx]
            mix_class_name = mix_wave_path.split(os.sep)[-2]
            mix_label = self.class_names_list.index(mix_class_name)
            mix_signal, mix_sample_rate = torchaudio.load(mix_wave_path)
            mix_audio = (mix_signal, mix_sample_rate)

            # 預處理另一段音訊
            mix_audio = self.resample(mix_audio, self.sample_rate)
            mix_audio = self.rechannel(mix_audio, self.channel)
            mix_audio = self.pad_trunc(mix_audio, self.duration)
            mix_signal, _ = mix_audio

            # mixup 的權重
            lam = np.random.beta(0.5, 0.5)
            # 混合 waveform
            signal = lam * signal + (1 - lam) * mix_signal

            # 混合 label（轉為 one-hot 向量）
            label_tensor = torch.zeros(len(self.class_names_list))
            label_tensor[label] = lam
            label_tensor[mix_label] = 1.0 - lam
        else:
            # 不進行 mixup，仍需 one-hot label
            label_tensor = torch.zeros(len(self.class_names_list))
            label_tensor[label] = 1.0

        # --- 音訊資料增強（在 mixup 之後做）---
        if self.augment and random.random() < 0.25:
            signal = self.random_gain(signal)
            signal = self.add_background_noise(signal)

        # 重建 audio tuple 給 spectrogram 使用
        audio = (signal, self.sample_rate)

        # 轉成 spectrogram
        spec = self.spectro_gram(audio, n_mels=64, n_fft=512, hop_len=None)

        # 再做 SpecAugment（選擇性）
        if self.augment and random.random() < 0.25:
            spec = self.spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return spec,audio[0],label_tensor
if __name__ == '__main__':
    train_path = '/home/zonekey/project/audio_classification/train'
    class_names_list = ['aloud', 'clap', 'discuss', 'noise', 'single']

    from torch.utils.data import WeightedRandomSampler
    from collections import Counter
    wave_path_list = []
    label_list = []

    for label_idx, class_name in enumerate(class_names_list):
        class_dir = os.path.join(train_path, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                wave_path_list.append(os.path.join(class_dir, filename))
                label_list.append(label_idx)  # 數字形式的 label

    label_counter = Counter(label_list)
    print("每類樣本數量:", label_counter)

    # 為每個樣本設置相對應的權重（樣本越少權重越高）
    sample_weights = [1.0 / label_counter[label] for label in label_list]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


    dataset = AudioDataset2(wave_path_list=wave_path_list, target_sample_rate=16000, duration=1000,
                           class_names_list=class_names_list,augment=True)
    train_loader = DataLoader(dataset=dataset, batch_size=16, sampler=sampler)
    for spec,signal,label in train_loader:
        print(signal.shape)
        print(spec.shape)
        print(label.shape)
        break

