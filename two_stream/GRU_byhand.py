import torch
import torch.nn as nn


def gru_forward_bidirectional(input, initial_state, w_ih, w_hh, b_ih, b_hh):
    """
    input: [B, T, E]
    initial_state: [2, B, H] (forward and backward initial states)
    w_ih, w_hh: [2, 3*H, E], [2, 3*H, H]
    b_ih, b_hh: [2, 3*H]
     FORMULA:
        rt=sigmoid(W_ir @ x + b_ir+ W_hr @ h + b_hr
        zt=sigmoid(W_iz@ x+ b_iz+ W_hz @ h + b_hz)
        nt=tanh(W_in @x +b_in +rt *(W_hn @ h + b_hn))
        ht= (1-zt)*nt+ zt*h
    return:
        output: [B, T, 2*H]
        h_n: [2, B, H]
    """
    B, T, E = input.shape
    D, H = 2, w_ih.shape[1] // 3  # 2 directions, hidden_size
    output_forward = torch.zeros(B, T, H)
    output_backward = torch.zeros(B, T, H)

    h_f = initial_state[0]  # [B, H]
    h_b = initial_state[1]  # [B, H]

    # Expand weights per batch
    batch_w_ih_f = w_ih[0].unsqueeze(0).expand(B, -1, -1)
    batch_w_hh_f = w_hh[0].unsqueeze(0).expand(B, -1, -1)
    batch_w_ih_b = w_ih[1].unsqueeze(0).expand(B, -1, -1)
    batch_w_hh_b = w_hh[1].unsqueeze(0).expand(B, -1, -1)

    # FORWARD DIRECTION
    for t in range(T):
        x = input[:, t, :]
        x_w = torch.bmm(batch_w_ih_f, x.unsqueeze(-1)).squeeze(-1)
        h_w = torch.bmm(batch_w_hh_f, h_f.unsqueeze(-1)).squeeze(-1)

        r_t = torch.sigmoid(x_w[:, :H] + h_w[:, :H] + b_ih[0, :H] + b_hh[0, :H])
        z_t = torch.sigmoid(x_w[:, H:2*H] + h_w[:, H:2*H] + b_ih[0, H:2*H] + b_hh[0, H:2*H])
        n_t = torch.tanh(x_w[:, 2*H:3*H] + b_ih[0, 2*H:3*H] + r_t * (h_w[:, 2*H:3*H] + b_hh[0, 2*H:3*H]))

        h_f = (1 - z_t) * n_t + z_t * h_f
        output_forward[:, t, :] = h_f

    # BACKWARD DIRECTION
    for t in reversed(range(T)):
        x = input[:, t, :]
        x_w = torch.bmm(batch_w_ih_b, x.unsqueeze(-1)).squeeze(-1)
        h_w = torch.bmm(batch_w_hh_b, h_b.unsqueeze(-1)).squeeze(-1)

        r_t = torch.sigmoid(x_w[:, :H] + h_w[:, :H] + b_ih[1, :H] + b_hh[1, :H])
        z_t = torch.sigmoid(x_w[:, H:2*H] + h_w[:, H:2*H] + b_ih[1, H:2*H] + b_hh[1, H:2*H])
        n_t = torch.tanh(x_w[:, 2*H:3*H] + b_ih[1, 2*H:3*H] + r_t * (h_w[:, 2*H:3*H] + b_hh[1, 2*H:3*H]))

        h_b = (1 - z_t) * n_t + z_t * h_b
        output_backward[:, t, :] = h_b

    # 拼接正向與反向結果
    output = torch.cat([output_forward, output_backward], dim=-1)  # [B, T, 2*H]
    h_n = torch.stack([h_f, h_b], dim=0)  # [2, B, H]

    return output, h_n

if __name__ == '__main__':

    # 測試參數
    bs, T, i_size, h_size = 2, 3, 4, 5
    input = torch.randn(bs, T, i_size)
    h0 = torch.randn(2, bs, h_size)

    # 建立雙向 GRU 模型
    gru_layer = nn.GRU(i_size, h_size, batch_first=True, bidirectional=True)
    output, h_final = gru_layer(input, h0)

    # 準備雙向參數
    w_ih = torch.stack([gru_layer.weight_ih_l0, gru_layer.weight_ih_l0_reverse])
    w_hh = torch.stack([gru_layer.weight_hh_l0, gru_layer.weight_hh_l0_reverse])
    b_ih = torch.stack([gru_layer.bias_ih_l0, gru_layer.bias_ih_l0_reverse])
    b_hh = torch.stack([gru_layer.bias_hh_l0, gru_layer.bias_hh_l0_reverse])

    # 調用自定義函數
    output_custom, h_final_custom = gru_forward_bidirectional(input, h0, w_ih, w_hh, b_ih, b_hh)

    # 比較
    print("output equal:", torch.allclose(output, output_custom, atol=1e-19))
    print("h_final equal:", torch.allclose(h_final, h_final_custom, atol=1e-19))