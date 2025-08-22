from model import PANNsCNN14Att
import torch
import pandas as pd
import numpy as np
from fastprogress import progress_bar
import soundfile as sf
class_names_list2 = ['aloud', 'clap', 'discuss', 'noise', 'single']
model_config = {
    "sample_rate": 16000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 5
}

def get_model(config: dict, weights_path: str):
    model = PANNsCNN14Att(**config)
    checkpoint = torch.load(weights_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model

weights_path = "train.3.pth"


def prediction_for_clip(clip: np.ndarray,
                        model: PANNsCNN14Att,sr,
                        threshold=0.5):
    PERIOD = 4  # 每段音频长度（秒）
    SR = sr
    total_duration = len(clip) / SR  # 计算音频总时长（秒）
    audios = []
    y = clip.astype(np.float32)
    start = 0
    end = PERIOD * SR

    # 分割音频
    while start < len(y):
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) < PERIOD * SR:
            y_pad = np.zeros(PERIOD * SR, dtype=np.float32)
            y_pad[:len(y_batch)] = y_batch
            audios.append(y_pad)
        else:
            audios.append(y_batch)
        start = end
        end += PERIOD * SR

    array = np.asarray(audios)
    tensors = torch.from_numpy(array)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    estimated_event_list = []
    global_time = 0.0  # 当前段的起始时间

    for i, image in enumerate(progress_bar(tensors)):
        image = image.view(1, image.size(0))
        image = image.to(device)

        with torch.no_grad():
            prediction = model(image)
            framewise_outputs = prediction["framewise_output"].detach().cpu().numpy()[0]

        thresholded = framewise_outputs >= threshold

        for target_idx in range(thresholded.shape[1]):
            if thresholded[:, target_idx].mean() == 0:
                continue  # 如果没有检测到事件，跳过

            detected = np.argwhere(thresholded[:, target_idx]).reshape(-1)
            head_idx = 0
            tail_idx = 0

            while head_idx < len(detected):
                if (tail_idx + 1 == len(detected)) or (detected[tail_idx + 1] - detected[tail_idx] != 1):
                    onset = 0.01 * detected[head_idx] + global_time
                    offset = 0.01 * detected[tail_idx] + global_time

                    # 确保时间戳不超过音频总长度
                    onset = min(onset, total_duration)
                    offset = min(offset, total_duration)

                    max_confidence = framewise_outputs[detected[head_idx]:detected[tail_idx] + 1, target_idx].max()
                    mean_confidence = framewise_outputs[detected[head_idx]:detected[tail_idx] + 1, target_idx].mean()

                    estimated_event = {
                        "category_code": class_names_list2[target_idx],
                        "onset": onset,
                        "offset": offset,
                        "max_confidence": max_confidence,
                        "mean_confidence": mean_confidence
                    }
                    estimated_event_list.append(estimated_event)
                    head_idx = tail_idx + 1
                    tail_idx = tail_idx + 1
                else:
                    tail_idx += 1

        global_time += PERIOD
        # 如果当前段是最后一段，且不足 PERIOD 秒，调整 global_time
        if (i == len(tensors) - 1) and (len(y_batch) < PERIOD * SR):
            global_time = total_duration  # 避免超出总时长

    prediction_df = pd.DataFrame(estimated_event_list)
    return prediction_df
if __name__ == '__main__':
    wave_path='../teacher_ac1.mp3'
    clip, sr = sf.read(wave_path)
    # change to single channel
    #clip=clip[:,0]
    print(clip.shape)
    model=get_model(model_config,weights_path)
    prediction_df=prediction_for_clip(clip,model,sr)
    prediction_df = prediction_df.sort_values(by="onset", ascending=True).reset_index(drop=True)
    with open('sed.txt', 'w') as f:
        for _,pred_row in prediction_df.iterrows():
            start_time=pred_row.loc['onset']
            end_time =pred_row.loc['offset']
            category = pred_row.loc['category_code']
            f.write(f"{start_time:.2f}\t{end_time:.2f}\t{category} \n \n")
