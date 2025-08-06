import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveformLogmelLongCNNClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(WaveformLogmelLongCNNClassifier, self).__init__()

        # ----- Waveform 分支 (Conv1D) -----
        self.wave_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=11, stride=5, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.ReLU(),
        )

        # ----- Logmel 分支 (Conv2D) -----
        self.logmel_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
        )

        # ----- 合併後的 CNN 層 -----
        self.fusion_cnn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ----- Pool + Classifier -----
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # For temporal global pooling
            nn.Flatten(),  # Flatten from (B, C, 1) → (B, C)
            nn.Linear(256, num_classes)  # *2 for bidirectional
        )


    def forward(self, waveform, logmel):
        # waveform: (B,channel=1， L)
        x_wave = self.wave_branch(waveform)  # → (B, C, T)
        x_wave = x_wave.unsqueeze(2)  # → (B, C, 1, T) 為了拼接時維度對齊

        # logmel:(B,  T, F)--> (B, 1, T, F)
        x_mel = self.logmel_branch(logmel)  # → (B, C, T1, F1)

        # Resize wave to match mel shape if needed
        x_wave = F.interpolate(x_wave, size=(x_mel.shape[2], x_mel.shape[3]), mode='bilinear')

        # Concat channel-wise
        x = torch.cat([x_wave, x_mel], dim=1)  # → (B, C_total=256, T, F)

        # Fusion CNN
        x = self.fusion_cnn(x)  # → (B, 512, T, F)
        x = x.mean(dim=3)  # → (B, 512, T) → 時序維度保留 (T)
        out = self.temporal_cnn(x)  # (B, 256, T)
        return self.classifier(out),x

class WaveformLogmelLongCNNClassifier2(nn.Module):
    def __init__(self, num_classes=5):
        super(WaveformLogmelLongCNNClassifier2, self).__init__()

        # ----- Waveform 分支 (Conv1D) -----
        self.wave_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,11), stride=(1,5), padding=(0,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(64, 128, kernel_size=(1,5), padding=(0,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(128, 256, kernel_size=(1,5), padding=(0,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),

            nn.Conv2d(256, 256, kernel_size=(1,5), padding=(0,2)),
            nn.ReLU(),
        )

        # ----- Logmel 分支 (Conv2D) -----
        self.logmel_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
        )

        # ----- 合併後的 CNN 層 -----
        self.fusion_cnn = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.temporal_cnn = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU()
        )

        # ----- Pool + Classifier -----
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # For temporal global pooling
            nn.Flatten(),  # Flatten from (B, C, 1) → (B, C)
            nn.Linear(256, num_classes)  # *2 for bidirectional
        )


    def forward(self, waveform, logmel):
        # waveform: (B,channel=1， L)
        x_wave = self.wave_branch(waveform)  # → (B, C, T)
        #x_wave = x_wave.unsqueeze(2)  # → (B, C, 1, T) 為了拼接時維度對齊

        # logmel:(B,  T, F)--> (B, 1, T, F)
        x_mel = self.logmel_branch(logmel)  # → (B, C, T1, F1)

        # Resize wave to match mel shape if needed
        x_wave = F.interpolate(x_wave, size=(x_mel.shape[2], x_mel.shape[3]), mode='bilinear')

        # Concat channel-wise
        x = torch.cat([x_wave, x_mel], dim=1)  # → (B, C_total=256, T, F)

        # Fusion CNN
        x = self.fusion_cnn(x)  # → (B, 512, T, F)
        x = x.mean(dim=3)  # → (B, 512, T) → 時序維度保留 (T)
        x = x.reshape((1,512,1,32))
        out = self.temporal_cnn(x)  # (B, 256, T)
        return self.classifier(out),x


if __name__ == '__main__':
    batch_size = 1
    waveform_len = 16000
    mel_T = 64
    mel_F = 63
    model1=WaveformLogmelLongCNNClassifier(num_classes=5)
    model2=WaveformLogmelLongCNNClassifier2(num_classes=5)
    save_path = './cnn512-2stream.pth'
    model1.load_state_dict(torch.load(save_path, map_location='cpu'), strict=True)
    model1.eval()
    # simulated data
    waveform = torch.randn(batch_size, 1, waveform_len, dtype=torch.float32)
    logmel = torch.randn(batch_size, 1, mel_T, mel_F, dtype=torch.float32)
    out1, _ = model1(waveform, logmel)

    state_dict = torch.load(save_path, map_location='cpu')
    # 將 Conv1d 權重升維到 Conv2d 形狀（[out_c, in_c, 1, k]）
    for key in list(state_dict.keys()):
        if "wave_branch" in key and "weight" in key:
            if state_dict[key].dim() == 3:
                state_dict[key] = state_dict[key].unsqueeze(2)  # [C_out, C_in, 1, K]
        elif "temporal_cnn" in key and "weight" in key:
            if state_dict[key].dim() == 3:
                state_dict[key] = state_dict[key].unsqueeze(2)  # [C_out, C_in, 1, K]
        else:
            continue
    model2.load_state_dict(state_dict, strict=True)
    model2.eval()
    out2, _ = model2(waveform.reshape((1,1,1,16000)), logmel)
    print(out1.shape)
    print(out2.shape)
    print("output equal:", torch.allclose(out1, out2, atol=1e-6))
    print("Max output diff:", (out1 - out2).abs().max())
    torch.onnx.export(
        model2,
        (waveform.reshape((1,1,1,16000)), logmel),  # 三個輸入
        "waveform_logmel_cnn2.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['waveform', 'logmel'],
        output_names=['output-1'],
        dynamic_axes=None
    )
    print("✅ ONNX 模型導出成功：waveform_logmel_cnn.onnx")



