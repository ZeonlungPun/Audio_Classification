import numpy as np
import torch


from YOHO_dataset import YOHODataset,construct_examples
from YOHO import YOHONet
rev_class_dict = {0: 'aloud', 1: 'clap', 2: 'discuss', 3: 'noise', 4: 'single'}
device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_class=len(rev_class_dict)
def mk_preds_YOHO(model, audio_path, no_of_div = 6, hop_size = 1.96,  win_length = 2.56,sampling_rate = 16000):

    model.eval().to(device)
    a, win_ranges = construct_examples(audio_path, win_len=win_length,hop_len=hop_size,sr=sampling_rate)

    #preds = np.zeros((len(a), no_of_div, 15))
    mss_in = torch.zeros((len(a), 64, 81))

    for i in range(len(a)):

        waveform=torch.tensor(a[i],dtype=torch.float32)
        M = YOHODataset.spectro_gram(waveform, n_mels=64, n_fft=1024, hop_len=512)
        mss_in[i, :, :] = M

    preds = model(mss_in.to(device))
    events = []

    for i in range(len(preds)):

        p = preds[i, :, :]
        events_curr = []
        win_width = win_length / no_of_div
        for j in range(len(p)):
            for jjj in range(0, num_class):
                if p[j][jjj*3] >= 0.5:

                    start = win_width * j + win_width * p[j][jjj*3+1] + win_ranges[i][0]
                    end = p[j][jjj*3+2] * win_width + start
                    events_curr.append([start, end, rev_class_dict[jjj]])

        events += events_curr


    class_set = set([c[2] for c in events])
    class_wise_events = {}

    for c in list(class_set):

        class_wise_events[c] = []


    for c in events:
        class_wise_events[c[2]].append(c)

    max_event_silence = 1.0
    all_events = []

    for k in list(class_wise_events.keys()):

        curr_events = class_wise_events[k]
        count = 0

        while count < len(curr_events) - 1:
          if (curr_events[count][1] >= curr_events[count + 1][0]) or (curr_events[count + 1][0] - curr_events[count][1] <= max_event_silence):
            curr_events[count][1] = max(curr_events[count + 1][1], curr_events[count][1])
            del curr_events[count + 1]
          else:
            count += 1

        all_events += curr_events

    for i in range(len(all_events)):

        all_events[i][0] = round(float(all_events[i][0].cpu().detach().numpy()), 3)
        all_events[i][1] = round(float(all_events[i][1].cpu().detach().numpy()), 3)

    all_events.sort(key=lambda x: x[0])

    return all_events

if __name__ == '__main__':
    model=YOHONet(num_classes=5)
    model_path='./model-best-error.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    audio_path='/home/zonekey/project/test/source5/test2.wav'
    all_events= mk_preds_YOHO(model,audio_path)
    print(all_events)
    with open('YOHO.txt','w') as f:
        for event in all_events:
            start_time=event[0]
            end_time =event[1]
            category=event[2]
            f.write(f"{start_time:.3f}\t{end_time:.3f}\t{category} \n")
