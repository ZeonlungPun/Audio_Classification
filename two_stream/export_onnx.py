import torch.onnx
from WaveformLogmel import WaveformLogmelCNNGRUClassifier

# 初始化模型
model = WaveformLogmelCNNGRUClassifier(num_classes=5)
save_path = './crnn-2stream.pth'
model.load_state_dict(torch.load(save_path, map_location='cpu'))
model.eval()

# Dummy inputs（根據你提供的 shape）
batch_size = 1
waveform_len = 16000
mel_T = 64
mel_F = 32
hidden_size = model.hidden_size
num_directions = model.num_directions

# 模擬一組輸入資料
waveform = torch.randn(batch_size, 1, waveform_len)
logmel = torch.randn(batch_size, 1, mel_T, mel_F)
hidden = torch.zeros(num_directions, batch_size, hidden_size, requires_grad=False)


# 導出模型
torch.onnx.export(
    model,
    (waveform, logmel, hidden),                      # 三個輸入
    "waveform_logmel_cnn_gru.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['waveform', 'logmel', 'hidden'],
    output_names=['output', 'hidden_out'],
    dynamic_axes=None
)

print("✅ ONNX 模型導出成功：waveform_logmel_cnn_gru.onnx")
