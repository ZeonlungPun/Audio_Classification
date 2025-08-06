import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUByHandBidirectional(nn.Module):
    def __init__(self, input_size, hidden_size, w_ih, w_hh, b_ih, b_hh):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Register parameters as buffers for ONNX compatibility
        self.register_buffer('w_ih_f', w_ih[0])
        self.register_buffer('w_hh_f', w_hh[0])
        self.register_buffer('b_ih_f', b_ih[0])
        self.register_buffer('b_hh_f', b_hh[0])
        self.register_buffer('w_ih_b', w_ih[1])
        self.register_buffer('w_hh_b', w_hh[1])
        self.register_buffer('b_ih_b', b_ih[1])
        self.register_buffer('b_hh_b', b_hh[1])

    def forward(self, x_t, h0): # x: [B, T, E], h0: [2, B, H]

        #B, T, E = 1, 32, 512
        H = self.hidden_size
        h_f = h0[0:1, :, :]
        h_b = h0[1:2, :, :]
        h_f =h_f.reshape((1,128))
        h_b =h_b.reshape((1,128))
        # [B, H]

        # Forward direction
        #t=1
        #x_t = x[:, t:(t+1), :]  # [B, E]

        x_t=x_t.reshape((1,512))
        gates_x = F.linear(x_t, self.w_ih_f, self.b_ih_f)  # [B, 3H]
        gates_h = F.linear(h_f, self.w_hh_f, self.b_hh_f)  # [B, 3H]
        #replace  r, z, n = gates_x.chunk(3, dim=1)
        chunk_size = 3*H // 3
        r = gates_x[:, 0:chunk_size]
        z = gates_x[:, chunk_size:2 * chunk_size]
        n = gates_x[:, 2 * chunk_size:3 * chunk_size]
        #replace r_h, z_h, n_h = gates_h.chunk(3, dim=1)
        r_h = gates_h[:, 0:chunk_size]
        z_h = gates_h[:, chunk_size:2 * chunk_size]
        n_h = gates_h[:, 2 * chunk_size:3 * chunk_size]

        r_t = torch.sigmoid(r + r_h)
        z_t = torch.sigmoid(z + z_h)
        target_shape = (1, 1, 1, 128)
        r_t = r_t.reshape(target_shape)
        z_t= z_t.reshape(target_shape)
        h_f= h_f.reshape(target_shape)
        n_h= n_h.reshape(target_shape)
        n_t = torch.tanh(n + r_t * n_h)
        n_t =n_t.reshape((target_shape))
        h_f = (1 - z_t) * n_t + z_t * h_f
        h_f=h_f.reshape((1, 128))

        return h_f



class GRUByHandBidirectionalF(nn.Module):
    def __init__(self, input_size, hidden_size, w_ih, w_hh, b_ih, b_hh):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 正向
        self.linear_ih_f = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.linear_hh_f = nn.Linear(hidden_size, 3 * hidden_size, bias=True)


        # 初始化權重
        with torch.no_grad():
            self.linear_ih_f.weight.copy_(w_ih[0])
            self.linear_hh_f.weight.copy_(w_hh[0])
            self.linear_ih_f.bias.copy_(b_ih[0])
            self.linear_hh_f.bias.copy_(b_hh[0])



    def for_content(self,x_t,h_f):
        #x_t (B,1,E)  h_f: [1, B, H]
        x_t = x_t.reshape((1, 512))
        h_f = h_f.reshape((1, 128))
        H = self.hidden_size
        gates_x = self.linear_ih_f(x_t)
        gates_h = self.linear_hh_f(h_f)
        # replace  r, z, n = gates_x.chunk(3, dim=1)
        chunk_size = 3 * H // 3
        r = gates_x[:, 0:chunk_size]
        z = gates_x[:, chunk_size:2 * chunk_size]
        n = gates_x[:, 2 * chunk_size:3 * chunk_size]
        # replace r_h, z_h, n_h = gates_h.chunk(3, dim=1)
        r_h = gates_h[:, 0:chunk_size]
        z_h = gates_h[:, chunk_size:2 * chunk_size]
        n_h = gates_h[:, 2 * chunk_size:3 * chunk_size]

        r_t = torch.sigmoid(r + r_h)
        z_t = torch.sigmoid(z + z_h)
        target_shape = (1,  1, 128)
        r_t = r_t.reshape(target_shape)
        z_t = z_t.reshape(target_shape)
        h_f = h_f.reshape(target_shape)
        n_h = n_h.reshape(target_shape)
        n_t = torch.tanh(n + r_t * n_h)
        n_t = n_t.reshape((target_shape))
        h_f = (1 - z_t) * n_t + z_t * h_f
        h_f = h_f.reshape((1, 128))

        return h_f


    def forward(self, x_t,h0): # x: [B, T=32, E=512], h0: [2, B, H]


        h_f1=self.for_content(x_t=x_t,h_f=h0)

        return h_f1



class GRUByHandBidirectionalB(nn.Module):
    def __init__(self, input_size, hidden_size, w_ih, w_hh, b_ih, b_hh):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 反向
        self.linear_ih_b = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.linear_hh_b = nn.Linear(hidden_size, 3 * hidden_size, bias=True)

        # 初始化權重
        with torch.no_grad():
            self.linear_ih_b.weight.copy_(w_ih[1])
            self.linear_hh_b.weight.copy_(w_hh[1])
            self.linear_ih_b.bias.copy_(b_ih[1])
            self.linear_hh_b.bias.copy_(b_hh[1])

    def for_content(self,x_t,h_b):
        # x_t (B,1,E)  h_f: [1, B, H]
        x_t = x_t.reshape((1, 512))
        h_b = h_b.reshape((1, 128))
        gates_x =  self.linear_ih_b(x_t)
        gates_h =  self.linear_hh_b(h_b)
        H = self.hidden_size
        # replace  r, z, n = gates_x.chunk(3, dim=1)
        chunk_size = 3 * H // 3
        r = gates_x[:, 0:chunk_size]
        z = gates_x[:, chunk_size:2 * chunk_size]
        n = gates_x[:, 2 * chunk_size:3 * chunk_size]
        # replace r_h, z_h, n_h = gates_h.chunk(3, dim=1)
        r_h = gates_h[:, 0:chunk_size]
        z_h = gates_h[:, chunk_size:2 * chunk_size]
        n_h = gates_h[:, 2 * chunk_size:3 * chunk_size]

        r_t = torch.sigmoid(r + r_h)
        z_t = torch.sigmoid(z + z_h)
        target_shape = (1,  1, 128)
        r_t = r_t.reshape(target_shape)
        z_t = z_t.reshape(target_shape)
        h_b = h_b.reshape(target_shape)
        n_h= n_h.reshape(target_shape)
        n_t = torch.tanh(n + r_t * n_h)
        h_b = (1 - z_t) * n_t + z_t * h_b
        h_b = h_b.reshape((1, 128))
        return h_b

    def forward(self,x_t,h0):

        h_b1 = self.for_content(x_t=x_t,h_b=h0)
        return h_b1



class WaveformLogmelCNNGRUClassifier(nn.Module):
    def __init__(self, num_classes=5,hidden_size=128):
        super(WaveformLogmelCNNGRUClassifier, self).__init__()
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = hidden_size
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
        # ----- GRU 時序建模層 -----
        self.gru = nn.GRU(input_size=512, hidden_size=self.hidden_size, num_layers=1,
                          batch_first=True, bidirectional=self.bidirectional)

        # ----- Pool + Classifier -----
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # For temporal global pooling
            nn.Flatten(),  # Flatten from (B, C, 1) → (B, C)
            nn.Linear(128 * 2, num_classes)  # *2 for bidirectional
        )
    def init_hidden(self, batch_size, device):
        # 建立初始全 0 隱藏狀態和 cell 狀態
        h0 = torch.zeros(
            self.num_directions, batch_size, self.hidden_size, device=device)

        return h0


    def forward(self, waveform, logmel,hidden=None):
        # waveform: (B,channel=1， L)
        x_wave = self.wave_branch(waveform)  # → (B, C, T)
        x_wave = x_wave.unsqueeze(2)  # → (B, C, 1, T) 為了拼接時維度對齊
        batch_size=x_wave.shape[0]
        device = x_wave.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # logmel:(B,  T, F)--> (B, 1, T, F)
        x_mel = self.logmel_branch(logmel)  # → (B, C, T1, F1)

        # Resize wave to match mel shape if needed
        x_wave = F.interpolate(x_wave, size=(x_mel.shape[2], x_mel.shape[3]), mode='bilinear')

        # Concat channel-wise
        x = torch.cat([x_wave, x_mel], dim=1)  # → (B, C_total=256, T, F)

        # Fusion CNN
        x = self.fusion_cnn(x)  # → (B, 512, T, F)
        x = x.mean(dim=3)  # → (B, 512, T) → 時序維度保留 (T)

        x = x.permute(0, 2, 1)  # → (B, T, 512)，適合 GRU 輸入

        out,  hidden = self.gru(x, hidden)  # → (B, T, hidden_size*2)
        out = out.permute(0, 2, 1)  # → (B, hidden_size*2, T)

        return self.classifier(out), hidden


def get_gru_params(gru_layer):
    # 準備雙向參數
    w_ih = torch.stack([gru_layer.weight_ih_l0, gru_layer.weight_ih_l0_reverse])
    w_hh = torch.stack([gru_layer.weight_hh_l0, gru_layer.weight_hh_l0_reverse])
    b_ih = torch.stack([gru_layer.bias_ih_l0, gru_layer.bias_ih_l0_reverse])
    b_hh = torch.stack([gru_layer.bias_hh_l0, gru_layer.bias_hh_l0_reverse])

    return w_ih,w_hh,b_ih,b_hh

def init_hidden( num_directions,hidden_size, batch_size, device):
    # 建立初始全 0 隱藏狀態和 cell 狀態
    h0 = torch.zeros(num_directions, batch_size, hidden_size, device=device)
    return h0


class WaveformLogmelCNNGRUClassifier2_1(nn.Module):
    def __init__(self):
        super(WaveformLogmelCNNGRUClassifier2_1, self).__init__()

        # ----- Waveform 分支 (Conv1D) -----
        # self.wave_branch = nn.Sequential(
        #     nn.Conv1d(1, 64, kernel_size=11, stride=5, padding=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=4),
        #
        #     nn.Conv1d(64, 128, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=4),
        #
        #     nn.Conv1d(128, 256, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=4),
        #
        #     nn.Conv1d(256, 256, kernel_size=5, padding=2),
        #     nn.ReLU(),
        # )
        self.wave_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 11), stride=(1, 5), padding=(0, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),

            nn.Conv2d(64, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),

            nn.Conv2d(128, 256, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),

            nn.Conv2d(256, 256, kernel_size=(1, 5), padding=(0, 2)),
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

    def forward(self, waveform, logmel):
        # waveform: (B,channel=1，1， L)
        x_wave = self.wave_branch(waveform)  # → (B, C, T)
        batch_size = 1
        #print(x_wave.shape)
        #x_wave = x_wave.reshape((batch_size,256,1,50))  # → (B, C, 1, T) 為了拼接時維度對齊

        # logmel:(B,  T, F)--> (B, 1, T, F)
        x_mel = self.logmel_branch(logmel)  # → (B, C, T1, F1)

        # Resize wave to match mel shape if needed
        x_wave = F.interpolate(x_wave, size=(x_mel.shape[2], x_mel.shape[3]), mode='bilinear')

        # Concat channel-wise
        x = torch.cat([x_wave, x_mel], dim=1)  # → (B, C_total=256, T, F)

        # Fusion CNN
        x = self.fusion_cnn(x)  # → (B, 512, T, F)
        x = x.mean(dim=3)  # → (B, 512, T) → 時序維度保留 (T)

        x = x.permute(0, 2, 1)  # → (B, T, 512)，適合 GRU 輸入

        return x

class WaveformLogmelCNNGRUClassifier2_2(nn.Module):
    def __init__(self,num_classes=5):
        super(WaveformLogmelCNNGRUClassifier2_2, self).__init__()
        # ----- Pool + Classifier -----
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # For temporal global pooling
            nn.Flatten(),  # Flatten from (B, C, 1) → (B, C)
            nn.Linear(128 * 2, num_classes)  # *2 for bidirectional
        )
        #self.n1=nn.AdaptiveAvgPool2d((1, 1))
        #self.n2=nn.Flatten()
        #self.n3=nn.Linear(128 * 2, num_classes)
    def forward(self,x):
        x=x.permute(0, 2, 1)  # → (B, hidden_size*2, T)
        x=x.reshape((1,256,1, 32))
        out=self.classifier(x)

        return out

class CNNGRUforHorizon(nn.Module):
    def __init__(self,w_ih, w_hh, b_ih, b_hh,model_path,hidden_size=128):
        super(CNNGRUforHorizon,self).__init__()
        self.save_path=model_path
        self.hidden_size=hidden_size
        self.cnn_layers=WaveformLogmelCNNGRUClassifier2_1()
        self.classification_layer=WaveformLogmelCNNGRUClassifier2_2(num_classes=5)
        self.load_parameters()
        self.grus_f = nn.Sequential()
        for i in range(32):
            self.grus_f.append(GRUByHandBidirectionalF(
                input_size=512,
                hidden_size=hidden_size,
                w_ih=w_ih,
                w_hh=w_hh,
                b_ih=b_ih,
                b_hh=b_hh))
        self.grus_b = nn.Sequential()
        for i in range(32):
            self.grus_b.append(GRUByHandBidirectionalB(
            input_size=512,
            hidden_size=hidden_size,
            w_ih=w_ih,
            w_hh=w_hh,
            b_ih=b_ih,
            b_hh=b_hh))


    def load_parameters(self):
        state_dict = torch.load(self.save_path, map_location='cpu')
        self.classification_layer.load_state_dict(state_dict,strict=False)
        # 將 Conv1d 權重升維到 Conv2d 形狀（[out_c, in_c, 1, k]）
        for key in list(state_dict.keys()):
            if "wave_branch" in key and "weight" in key:
                if state_dict[key].dim() == 3:
                    state_dict[key] = state_dict[key].unsqueeze(2)  # [C_out, C_in, 1, K]
        self.cnn_layers.load_state_dict(state_dict,strict=False)

    def forward(self,waveform,logmel):
        x = self.cnn_layers(waveform.reshape((1, 1, 1, 16000)), logmel)
        hidden = torch.zeros(2, 1, self.hidden_size, requires_grad=False).contiguous()
        hf0 = hidden[0:1, :, :]
        hb0 = hidden[1:2, :, :]
        # gru forward
        output_f=[]
        output_b = []
        hf = hf0
        for i in range(32):
            hf = self.grus_f[i](x_t=x[:, i:i+1, :], h0=hf)
            output_f.append(hf)
        output_f=torch.stack(output_f,dim=1)
        hb= hb0
        for j in reversed(range(32)):
            hb= self.grus_b[j](x_t=x[:, j:j+1, :], h0=hb)
            output_b.append(hb)
        output_b = torch.stack(output_b, dim=1)

        output_gru = torch.cat([output_f, output_b], dim=-1)  # [B, T, 2H]
        logits= self.classification_layer(output_gru)

        return logits


if __name__ == '__main__':
    batch_size = 1
    waveform_len = 16000
    mel_T = 64
    mel_F = 63
    hidden_size = 128
    num_directions = 2

    # simulated data
    waveform = torch.randn(batch_size, 1, waveform_len,dtype=torch.float32)
    logmel = torch.randn(batch_size, 1, mel_T, mel_F,dtype=torch.float32)
    hidden = torch.zeros(num_directions, batch_size, hidden_size, requires_grad=False).contiguous()
    # the process of model 1
    model1=WaveformLogmelCNNGRUClassifier(num_classes=5)
    save_path = './crnn-2stream.pth'
    model1.load_state_dict(torch.load(save_path, map_location='cpu'),strict=True)
    model1.eval()
    out1,_= model1(waveform,logmel)

    #  the process of model 2
    gru_layer =model1.gru
    w_ih, w_hh, b_ih, b_hh = get_gru_params(gru_layer)
    model2=CNNGRUforHorizon(w_ih, w_hh, b_ih, b_hh,'./crnn-2stream.pth')
    model2.eval()
    out2 = model2(waveform.reshape((1,1,1,16000)), logmel)
    print(out1.shape)
    print(out2.shape)
    print("output equal:", torch.allclose(out1, out2, atol=1e-6))
    print("Max output diff:", (out1 - out2).abs().max())
    torch.onnx.export(
        model2,
        (waveform.reshape((1,1,1,16000)), logmel),  # 三個輸入
        "waveform_logmel_CNN_GRU.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['waveform', 'logmel'],
        output_names=['output-logits'],
        dynamic_axes=None
    )
    print("✅ ONNX 模型導出成功")
    # model2_1=WaveformLogmelCNNGRUClassifier2_1()
    # # 載入舊模型的 state_dict（Conv1d 格式）
    # state_dict = torch.load(save_path, map_location='cpu')
    # # 將 Conv1d 權重升維到 Conv2d 形狀（[out_c, in_c, 1, k]）
    # for key in list(state_dict.keys()):
    #     if "wave_branch" in key and "weight" in key:
    #         if state_dict[key].dim() == 3:
    #             state_dict[key] = state_dict[key].unsqueeze(2)  # [C_out, C_in, 1, K]
    #
    # # 載入到新模型（Conv2d 架構）
    # model2_1.load_state_dict(state_dict, strict=False)
    # model2_1.eval()
    # model2_2= WaveformLogmelCNNGRUClassifier2_2(num_classes=5)
    # model2_2.load_state_dict(torch.load(save_path, map_location='cpu'), strict=False)
    # model2_2.eval()

    # gru_custom_f = GRUByHandBidirectionalF(
    #         input_size=512,
    #         hidden_size=hidden_size,
    #         w_ih=w_ih,
    #         w_hh=w_hh,
    #         b_ih=b_ih,
    #         b_hh=b_hh)
    # gru_custom_f.eval()
    # gru_custom_b = GRUByHandBidirectionalB(
    #     input_size=512,
    #     hidden_size=hidden_size,
    #     w_ih=w_ih,
    #     w_hh=w_hh,
    #     b_ih=b_ih,
    #     b_hh=b_hh)
    # gru_custom_b.eval()
    # x=model2_1(waveform.reshape((1,1,1,16000)),logmel)
    # output_f, output_b=[],[]
    # T=32
    # hf = hidden[0:1, :, :]
    # hb = hidden[1:2, :, :]
    # # forward direction
    # for t in range(T):
    #     x_t = x[:, t:(t+1), :]  # [B, E]
    #     hf=gru_custom_f(x_t=x_t,h0=hf)
    #     output_f.append(hf)
    # # backward direction
    # for t in reversed(range(T)):
    #     x_t = x[:, t:(t+1), :]  # [B, E]
    #     hb=gru_custom_b(x=x,h0=hb)
    #     output_b.append(hb)
    # output_f = torch.stack(output_f, dim=1)
    # output_b = torch.stack(output_b, dim=1)
    # # 拼接正反向結果
    # output_gru = torch.cat([output_f, output_b], dim=-1)  # [B, T, 2H]
    # h_n = torch.stack([hf, hb], dim=0)  # [2, B, H]
    # out2 = model2_2(output_gru)


    # torch.onnx.export(
    #     model2_1,
    #     (waveform.reshape((1,1,1,16000)), logmel),  # 三個輸入
    #     "waveform_logmel_cnn_gru2_1.onnx",
    #     export_params=True,
    #     opset_version=11,
    #     do_constant_folding=True,
    #     input_names=['waveform', 'logmel'],
    #     output_names=['output-1'],
    #     dynamic_axes=None
    # )
    # print("✅ ONNX 模型導出成功：waveform_logmel_cnn_gru2_1.onnx")

    # model2_2_input=torch.randn(*output_gru.shape,dtype=torch.float32)
    # torch.onnx.export(
    #     model2_2,
    #     model2_2_input,
    #     "waveform_logmel_cnn_gru2_2.onnx",
    #     export_params=True,
    #     opset_version=11,
    #     do_constant_folding=True,
    #     input_names=['woutput_gru'],
    #     output_names=['output'],
    #     dynamic_axes=None
    # )
    # print("✅ ONNX 模型導出成功：waveform_logmel_cnn_gru2_2.onnx")
    # x_t = torch.randn(1, 1, 512)  # 一個時間步
    # h_t = torch.randn(1,1, 128)
    # torch.onnx.export(
    #     gru_custom_f,
    #     (x_t, h_t),
    #     "gru_custom_f.onnx",
    #     input_names=['x_t_f', 'h_t_f'],
    #     output_names=['hf'],
    #     opset_version=11,
    #     do_constant_folding=True,
    #     dynamic_axes=None
    # )
    # print("✅ ONNX 模型導出成功：gru_custom_f.onnx")
    # torch.onnx.export(
    #     gru_custom_b,
    #     (x_t, h_t),
    #     "gru_custom_b.onnx",
    #     input_names=['x_b', 'h_b'],
    #     output_names=['hb'],
    #     opset_version=11,
    #     do_constant_folding=True,
    #     dynamic_axes=None
    # )
    # print("✅ ONNX 模型導出成功：gru_custom_b.onnx")


    # hidden_size = 128
    # input_size = 512
    # batch_size = 1
    # seq_len = 32
    #
    # # 建立一個 PyTorch GRU 並取得參數
    # torch_gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
    # w_ih = torch.stack([torch_gru.weight_ih_l0, torch_gru.weight_ih_l0_reverse])
    # w_hh = torch.stack([torch_gru.weight_hh_l0, torch_gru.weight_hh_l0_reverse])
    # b_ih = torch.stack([torch_gru.bias_ih_l0, torch_gru.bias_ih_l0_reverse])
    # b_hh = torch.stack([torch_gru.bias_hh_l0, torch_gru.bias_hh_l0_reverse])
    #
    # # 建立 GRUByHandBidirectional 實例
    # gru_model = GRUByHandBidirectional(input_size, hidden_size, w_ih, w_hh, b_ih, b_hh)
    # gru_model.eval()
    #
    # # 假資料
    # #x = torch.randn(batch_size, seq_len, input_size)  # [B, T, E]
    # x_t = torch.randn(batch_size, 1, input_size)  # [B, T, E]
    # h0 = torch.zeros(2, batch_size, hidden_size)  # [2, B, H]
    # t=2
    # # 匯出 ONNX
    # torch.onnx.export(
    #     gru_model,
    #     (x_t, h0),
    #     "gru_by_hand_bidirectional.onnx",
    #     export_params=True,
    #     opset_version=11,
    #     input_names=['x', 'h0'],
    #     output_names=['h_n'],
    #     dynamic_axes=None
    # )
    #
    # print("✅ 成功導出 GRUByHandBidirectional 至 gru_by_hand_bidirectional.onnx")