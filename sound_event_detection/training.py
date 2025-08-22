import torch,os,glob,sed_eval,dcase_util
import torch.optim as optim
from YOHO import YOHONet
from YOHO_dataset import YOHODataset


def mk_preds_YOHO_mel(model,input ,win_ranges, rev_class_dict,
                             no_of_div=6,win_length=2.56,
                             max_event_silence=0.3):


    with torch.no_grad():
                                         # input : (N, 1, 257, 40)
        preds = model(input)             # output: (N, T, 18) for 6 classes

    preds = preds.cpu().numpy()
    events = []

    for i in range(len(preds)):  # 每個 window
        p = preds[i]  # shape (no_div, num_class*3)
        events_curr = []
        win_width = win_length / no_of_div
        win_start = win_ranges[0][i]

        for j in range(len(p)):  # time bin index
            for cls in range(len(rev_class_dict)):
                base = cls * 3
                if p[j][base] >= 0.5:
                    start = win_width * j + win_width * p[j][base + 1] + win_start
                    end = p[j][base + 2] * win_width + start
                    events_curr.append([start, end, rev_class_dict[cls]])

        events += events_curr

    # 合併重疊事件
    class_wise_events = {}
    for e in events:
        class_wise_events.setdefault(e[2], []).append(e)

    all_events = []
    for cls, evts in class_wise_events.items():
        evts.sort(key=lambda x: x[0])
        count = 0
        while count < len(evts) - 1:
            if (evts[count][1] >= evts[count + 1][0]) or (evts[count + 1][0] - evts[count][1] <= max_event_silence):
                evts[count][1] = max(evts[count][1], evts[count + 1][1])
                del evts[count + 1]
            else:
                count += 1
        all_events.extend(evts)

    # 四捨五入 + 排序
    for e in all_events:
        e[0] = round(float(e[0].numpy()), 3)
        e[1] = round(float(e[1].numpy()), 3)
    all_events.sort(key=lambda x: x[0])

    return all_events


def my_loss_fn(y_true, y_pred):
    """
    y_true: Tensor of shape (B, no_of_div, num_class*3)
    y_pred: Tensor of shape (B, no_of_div, num_class*3)
    """

    squared_difference = (y_true - y_pred) ** 2  # shape: (B, no_of_div, num_class*3)

    # 將每類別的 presence mask 抽出
    ss_True = torch.ones_like(y_true[:, :, 0])  # shape: (B, no_of_div)
    ss_0 = y_true[:, :, 0]
    ss_1 = y_true[:, :, 3]
    ss_2 = y_true[:, :, 6]
    ss_3 = y_true[:, :, 9]
    ss_4 = y_true[:, :, 12]

    # 建構 mask：對每個類別 [1, presence, presence]
    sss = torch.stack([
        ss_True, ss_0, ss_0,
        ss_True, ss_1, ss_1,
        ss_True, ss_2, ss_2,
        ss_True, ss_3, ss_3,
        ss_True, ss_4, ss_4
    ], dim=2)  # shape: (B, no_of_div, num_class*3)

    # 僅對需要訓練的項目計算損失
    masked_loss = squared_difference * sss

    return masked_loss.sum(dim=[1, 2]).mean()

def evaluate_model_from_loader(model, val_loader, rev_class_dict,
                               pred_dir, device='cuda'):

    model.to(device).eval()
    os.makedirs(pred_dir, exist_ok=True)

    all_event_labels = set()
    data = []

    for batch_idx, (inputs, labels_array_tensor, label_full_path, w_range) in enumerate(val_loader):
        inputs = inputs.to(device)
        all_events = mk_preds_YOHO_mel(model, inputs, w_range, rev_class_dict)

        # 生成預測檔案路徑
        pred_file = os.path.join(
            pred_dir,
            os.path.basename(label_full_path[batch_idx]).replace(".txt", "-se-prediction.ann")
        )

        # 寫入預測結果
        with open(pred_file, 'w') as f:
            for e in all_events:
                f.write(f"{e[0]},{e[1]},{e[2]}\n")

        # 載入 ground truth 和預測
        gt_path = label_full_path[batch_idx].replace(".txt", ".ann").replace('label', 'label_ann')
        gt = dcase_util.containers.MetaDataContainer().load(gt_path)
        est = dcase_util.containers.MetaDataContainer().load(pred_file)

        # 收集所有類別標籤
        all_event_labels.update(event['event_label'] for event in gt)
        all_event_labels.update(event['event_label'] for event in est)

        data.append({
            'reference_event_list': gt,
            'estimated_event_list': est
        })

    event_labels = sorted(list(all_event_labels))

    # 計算評估指標
    segment_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=event_labels, time_resolution=1.0)
    event_metrics = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=event_labels, t_collar=1.0)

    for pair in data:
        segment_metrics.evaluate(pair['reference_event_list'], pair['estimated_event_list'])
        event_metrics.evaluate(pair['reference_event_list'], pair['estimated_event_list'])

    results = segment_metrics.results_overall_metrics()
    f1_val = results['f_measure']['f_measure']
    error_val = results['error_rate']['error_rate']

    print(segment_metrics)
    print(event_metrics)

    return f1_val, error_val

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10,
                device='cuda', resume_path='model-best-f1.pth', resume_metric='f1'):
    """
    resume_path: str or None, e.g., "model-best-f1.pth"
    resume_metric: 'f1' or 'error' — 用來設定 resume 模型的 best_xx 初始值
    """
    best_f1 = 0.0
    best_error = float('inf')

    model.to(device)

    # Resume if needed
    if resume_path and os.path.isfile(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"✅ Loaded model weights from {resume_path}")
        if resume_metric == 'f1':
            best_f1 = 1e-6  # 假設已有一個可提升的 F1（因為會在第10 epoch開始評估）
        elif resume_metric == 'error':
            best_error = 1.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}')

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            rev_class_dict = {0: 'aloud', 1: 'clap', 2: 'discuss', 3: 'noise', 4: 'single'}
            f1_val, error_val = evaluate_model_from_loader(
                model, val_loader, rev_class_dict, './logs', device
            )

            print(f'Validation F1: {f1_val:.4f}, Error Rate: {error_val:.4f}')

            # Save model if improved
            if f1_val > best_f1:
                best_f1 = f1_val
                torch.save(model.state_dict(), "model-best-f1.pth")
                print(f"📦 Saved best F1 model: {best_f1:.4f}")

            if error_val < best_error:
                best_error = error_val
                torch.save(model.state_dict(), "model-best-error.pth")
                print(f"📦 Saved best Error Rate model: {best_error:.4f}")

    print(f"🏁 Best F1 score: {best_f1:.4f}")
    print(f"🏁 Best error rate: {best_error:.4f}")
if __name__ == '__main__':
    class_list= ['aloud', 'clap', 'discuss', 'noise', 'single']
    device ="cuda:0" if torch.cuda.is_available() else "cpu"
    train_dir='/home/zonekey/project/audio_classification/sound_event_detection/synthetic_dataset/train'
    val_dir='/home/zonekey/project/audio_classification/sound_event_detection/synthetic_dataset/val'

    # Initialize model, dataset, and dataloaders
    model = YOHONet(num_classes=5)
    train_dataset = YOHODataset(train_dir,  class_list,augment=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
    val_dataset = YOHODataset(val_dir, class_list,augment=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)
    optimizer= optim.Adam(model.parameters(), lr=0.0001)
    train_model(model,train_loader,val_loader,criterion=my_loss_fn,optimizer=optimizer,
                num_epochs=3500,device=device)
