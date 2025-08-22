import os,random
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np


class SEDDataset(Dataset):
    def __init__(self,wave_path_list,class_names_list,waveform_transforms=None,augment=False):
        self.wave_path_list = wave_path_list
        self.class_names_list = class_names_list
        self.waveform_transforms=waveform_transforms
        self.augment=augment
    def __len__(self):
        return len(self.wave_path_list)

    def _pad_or_crop(self, y, sr, period):
        effective_length = sr * period
        len_y = len(y)
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            return new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            return y[start:start + effective_length].astype(np.float32)
        else:
            return y.astype(np.float32)

    def __getitem__(self, idx):
        # 基本加載
        wave_path = self.wave_path_list[idx]
        class_name = wave_path.split('/')[-2]
        label = self.class_names_list.index(class_name)

        PERIOD = 2
        y, sr = sf.read(wave_path)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            y = self._pad_or_crop(y, sr, PERIOD)

        # one-hot label
        labels = np.zeros(len(self.class_names_list), dtype="f")
        labels[label] = 1

        # 是否進行 mixup
        if self.augment and np.random.rand() < 0.65:
            # 隨機選取第二段音訊
            mix_idx = random.randint(0, len(self.wave_path_list) - 1)
            mix_path = self.wave_path_list[mix_idx]
            mix_class = mix_path.split('/')[-2]
            mix_label = self.class_names_list.index(mix_class)
            y2, sr2 = sf.read(mix_path)
            y2 = self.waveform_transforms(y2) if self.waveform_transforms else self._pad_or_crop(y2, sr2, PERIOD)


            alpha = 0.4  # 預設值

            lam = np.random.beta(alpha, alpha)

            # 混合 waveform
            y = lam * y + (1 - lam) * y2

            # 混合 label
            labels_mix = np.zeros(len(self.class_names_list), dtype="f")
            labels_mix[label] = lam
            labels_mix[mix_label] = 1.0 - lam
            labels = labels_mix

        return {"waveform": y, "targets": labels}


if __name__ == '__main__':
    train_path = '/home/zonekey/project/audio_classification/train'
    class_names_list = ['aloud_manual', 'clap', 'discuss_manual', 'noise', 'single_manual']
    wave_path_list = []


    for label_idx, class_name in enumerate(class_names_list):
        class_dir = os.path.join(train_path, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                wave_path_list.append(os.path.join(class_dir, filename))

    dataset = SEDDataset(wave_path_list=wave_path_list,
                           class_names_list=class_names_list)
    train_loader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

    for batch in train_loader:
        waveforms = batch["waveform"]  # Shape: [batch_size, 5 * sr]
        targets = batch["targets"]  # Shape: [batch_size, num_classes]
        print(waveforms.shape)
        print(targets.shape)
        break


