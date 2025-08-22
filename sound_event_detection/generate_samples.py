import os
import random
import torchaudio
import torch

def generate_synthetic_dataset(
        input_root,
        output_dir,
        class_names_list,
        sample_rate=16000,
        min_segments=3,
        max_segments=5,
        max_total_duration=2.5,
        min_clip_duration=0.5,
        max_clip_duration=1.2,
        total_samples=100
        , probabilities = [0.2, 0.2, 0.2 , 0.2, 0.2]):
    os.makedirs(output_dir, exist_ok=True)
    wav_dir = os.path.join(output_dir, "wav")
    label_dir = os.path.join(output_dir, "label")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # 收集各類別音訊
    class_dirs = sorted([
        os.path.join(input_root, d)
        for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])
    class_to_idx = {d: i for i, d in enumerate(class_names_list)}
    class_samples = {
        cls: [os.path.join(cls_path, f)
              for f in os.listdir(cls_path)
              if f.endswith(".wav")]
        for cls_path in class_dirs
        for cls in [os.path.basename(cls_path)]
    }

    for idx in range(total_samples):
        concat_wave = []
        event_log = []
        cur_time = 0.0

        while True:
            cls = random.choices(list(class_samples.keys()),weights=probabilities,k=1)[0]
            path = random.choice(class_samples[cls])
            wave, sr = torchaudio.load(path)

            if sr != sample_rate:
                wave = torchaudio.functional.resample(wave, sr, sample_rate)


            total_len_sec = wave.shape[1] / sample_rate
            max_crop_dur = min(max_clip_duration, total_len_sec)
            crop_duration = random.uniform(min_clip_duration, max_crop_dur)
            crop_len = int(crop_duration * sample_rate)
            if wave.shape[1] < crop_len:
                continue  # 音檔太短，跳過

            start_sample = random.randint(0, wave.shape[1] - crop_len)
            crop = wave[:, start_sample:start_sample + crop_len]

            if max_crop_dur < min_clip_duration:
                continue  # 略過太短的音檔

            # 加上這段會超過 2.5 秒就跳出
            if cur_time + crop_duration > max_total_duration:
                break

            event_log.append(f"{class_to_idx[cls]} {cur_time:.3f} {cur_time + crop_duration:.3f}")
            concat_wave.append(crop)
            cur_time += crop_duration

            # 至少保證最少片段數
            if len(event_log) >= max_segments:
                break

        # 若片段數 < min_segments，則重新取樣
        if len(event_log) < min_segments:
            continue

        # 拼接並保存
        new_wave = torch.cat(concat_wave, dim=1)
        wav_path = os.path.join(wav_dir, f"sample_{idx:04d}.wav")
        label_path = os.path.join(label_dir, f"sample_{idx:04d}.txt")

        torchaudio.save(wav_path, new_wave, sample_rate)
        with open(label_path, "w") as f:
            f.write("\n".join(event_log))

        print(f"✅ Saved: {wav_path} - Duration: {cur_time:.2f}s, Events: {len(event_log)}")


if __name__ == "__main__":
    class_names_list = ['aloud', 'clap', 'discuss', 'noise', 'single']
    generate_synthetic_dataset(
        input_root="/home/zonekey/project/audio_classification/clean_dataset/val_manual",
        output_dir="./synthetic_dataset/val",
        total_samples=1000,class_names_list=class_names_list)