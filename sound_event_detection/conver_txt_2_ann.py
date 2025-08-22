import os

def convert_txt_to_ann(txt_path, ann_path, class_list):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    ann_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            continue  # skip malformed lines
        cls_idx, start, end = int(parts[0]), float(parts[1]), float(parts[2])
        class_name = class_list[cls_idx]
        ann_line = f"{start:.3f},{end:.3f},{class_name}"
        ann_lines.append(ann_line)

    with open(ann_path, 'w') as f:
        f.write('\n'.join(ann_lines))

class_list = ['aloud', 'clap', 'discuss', 'noise', 'single']

txt_dir = "/home/zonekey/project/audio_classification/sound_event_detection/synthetic_dataset/val/label"
ann_dir = "/home/zonekey/project/audio_classification/sound_event_detection/synthetic_dataset/val/label_ann"
os.makedirs(ann_dir, exist_ok=True)

for txt_file in os.listdir(txt_dir):
    if txt_file.endswith(".txt"):
        txt_path = os.path.join(txt_dir, txt_file)
        ann_path = os.path.join(ann_dir, txt_file.replace(".txt", ".ann"))
        convert_txt_to_ann(txt_path, ann_path, class_list)
