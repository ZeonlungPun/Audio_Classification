from WaveformLogmel import WaveformLogmelClassifier,WaveformLogmelCNNGRUClassifier,LogmelCNNGRUClassifier,WaveformLogmelLongCNNClassifier
from WaveformLogmelDatasets import AudioDataset,AudioDataset2
from torch.utils.data import WeightedRandomSampler
from collections import Counter, defaultdict
import os,torch
from torch.utils.data import  DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn

def inference(model, val_dl, class_names,device,using_hidden=None):
    model.eval()
    correct_prediction = 0
    total_prediction = 0
    all_preds = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        hidden=None
        for spec,signal, labels in tqdm(val_dl, desc="Evaluating", leave=False):
            spec, signal, labels = spec.to(device),signal.to(device) ,labels.to(device)
            # Forward
            if using_hidden:
                if hidden is not None and hidden[0].size(1) != signal.size(0):
                    hidden = None
                outputs, hidden = model(signal, spec, hidden)
                if hidden is not None:
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.detach() for h in hidden)
                    else:
                        hidden = hidden.detach()
            else:
                outputs, _ = model(signal, spec)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 統計整體正確率
            correct_prediction += (preds == labels).sum().item()
            total_prediction += preds.size(0)

            # 統計分類正確率
            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if label == pred:
                    class_correct[label.item()] += 1

    # 整體正確率
    acc = correct_prediction / total_prediction
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'\n✅ Overall Accuracy: {acc:.2f}, Macro-F1: {macro_f1:.2f}')
    print(f'Total items: {total_prediction}')
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f'Confusion Matrix (Acc: {acc:.4f}, F1: {macro_f1:.4f})')
    plt.tight_layout()
    plt.savefig('cm.png')

    # 每個類別的正確率
    print('\n📊 Per-Class Accuracy:')
    for i, class_name in enumerate(class_names):
        total = class_total[i]
        correct = class_correct[i]
        acc = correct / total if total > 0 else 0.0
        print(f'  {class_name}: {acc:.2f} ({correct}/{total})')
    return macro_f1
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs, val_dl, class_names1,class_names2,save_path='best.pth',one_hot_label=False,using_hidden=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights=torch.tensor([1.05,1,1,0.8,1.25]).to(device)
    if one_hot_label:
        criterion =nn.BCEWithLogitsLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001,
        steps_per_epoch=len(train_dl),
        epochs=num_epochs,
        anneal_strategy='linear'
    )
    best_macro_f1 = 0.0  # ← 初始化最佳驗證準確率

    # 🔁 如果有已保存的模型，讀取模型參數並載入
    if os.path.exists(save_path):
        print(f"📥 Loading existing model from '{save_path}'...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.to(device)
        # 一次驗證，更新 best_macro_f1：
        if val_dl is not None:
            best_macro_f1 = inference(model, val_dl, class_names1,device,using_hidden)
            print(f"🔍 Loaded model macro-F1: {best_macro_f1:.4f}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
        hidden = None
        # 🔄 使用 tqdm 顯示每個 epoch 的進度
        loop = tqdm(train_dl, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for spec,signal, labels in loop:
            print(spec.shape)
            print(signal.shape)
            spec,signal, labels = spec.to(device),signal.to(device) ,labels.to(device)
            model.to(device)

            optimizer.zero_grad()
            if using_hidden:
                if hidden is not None and hidden[0].size(1) != signal.size(0):
                    hidden = None
                outputs, hidden = model(signal, spec, hidden)
                if hidden is not None:
                    if isinstance(hidden, tuple):
                        hidden = tuple(h.detach() for h in hidden)
                    else:
                        hidden = hidden.detach()
            else:
                outputs, _ = model(signal, spec)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)

            if one_hot_label:
                _,labels= torch.max(labels,1)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            # 🟦 更新 tqdm 顯示內容
            loop.set_postfix(loss=loss.item(), acc=correct_prediction / total_prediction)

        # 每個 epoch 結束後列印結果
        epoch_loss = running_loss / len(train_dl)
        epoch_acc = correct_prediction / total_prediction
        print(f"✅ Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

        # 每幾個 epoch 呼叫一次驗證函數
        if val_dl is not None and ( (epoch+1) %5==0 or epoch==num_epochs):
            macro_f1=inference(model, val_dl, class_names2,device,using_hidden)
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), save_path)
                print(f'🎯 Best model updated and saved (Macro-F1: {macro_f1:.4f})')


if __name__ == '__main__':

    train_path = '/home/zonekey/project/audio_classification/train'
    train_path2= '/home/zonekey/project/audio_classification/train'
    class_names_list1 = [ 'clap']
    class_names_list1_ = ['aloud', 'discuss', 'noise', 'single']
    val_path = '/home/zonekey/project/audio_classification/clean_dataset/val_manual'
    val_path2 = '/home/zonekey/project/audio_classification/clean_dataset/val_manual'
    class_names_list2 = ['clap']
    class_names_list2_ = ['aloud', 'discuss', 'noise', 'single']
    class_names_list = ['aloud','clap', 'discuss', 'noise', 'single']
    wave_path_list1 = []
    label_list1 = []
    wave_path_list2 = []
    label_list2 = []

    for label_idx, class_name in enumerate(class_names_list1):
        class_dir = os.path.join(train_path, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                wave_path_list1.append(os.path.join(class_dir, filename))
                label_list1.append(class_names_list.index(class_name))  # 數字形式的 label

    for label_idx, class_name in enumerate(class_names_list1_):
        class_dir = os.path.join(train_path2, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                wave_path_list1.append(os.path.join(class_dir, filename))
                label_list1.append(class_names_list.index(class_name))  # 數字形式的 label

    label_counter1 = Counter(label_list1)
    print("每類樣本數量:", label_counter1)

    # 為每個樣本設置相對應的權重（樣本越少權重越高）
    sample_weights1 = [1.0 / label_counter1[label] for label in label_list1]

    for label_idx, class_name in enumerate(class_names_list2):
        class_dir = os.path.join(val_path, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                wave_path_list2.append(os.path.join(class_dir, filename))
                label_list2.append(class_names_list.index(class_name))  # 數字形式的 label

    for label_idx, class_name in enumerate(class_names_list2_):
        class_dir = os.path.join(val_path2, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                wave_path_list2.append(os.path.join(class_dir, filename))
                label_list2.append(class_names_list.index(class_name))  # 數字形式的 label

    label_counter2 = Counter(label_list2)
    print("每類樣本數量:", label_counter2)

    # 為每個樣本設置相對應的權重（樣本越少權重越高）
    sample_weights2 = [1.0 / label_counter2[label] for label in label_list2]

    sampler1 = WeightedRandomSampler(
        weights=sample_weights1,
        num_samples=len(sample_weights1),
        replacement=True
    )

    train_dataset = AudioDataset2(wave_path_list=wave_path_list1, target_sample_rate=16000, duration=1000,
                           class_names_list=class_names_list,target_channel=1,augment=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler1)
    sampler2 = WeightedRandomSampler(
        weights=sample_weights2,
        num_samples=len(sample_weights2),
        replacement=True
    )

    val_dataset = AudioDataset(wave_path_list=wave_path_list2, target_sample_rate=16000, duration=1000,
                           class_names_list=class_names_list,target_channel=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16, sampler=sampler2)

    #model = WaveformLogmelClassifier(num_classes=5)
    #model =WaveformLogmelCNNGRUClassifier(num_classes=5)
    #model =LogmelCNNGRUClassifier(num_classes=5)
    model = WaveformLogmelLongCNNClassifier(num_classes=5)
    num_epochs = 50
    training(model, train_loader, num_epochs, val_loader, class_names_list,class_names_list,save_path='cnn512-2stream-test.pth',one_hot_label=True,using_hidden=False)











