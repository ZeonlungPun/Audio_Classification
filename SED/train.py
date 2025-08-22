import logging,time,os,torch,warnings
from typing import Optional, List
from contextlib import contextmanager
from model import PANNsCNN14Att
from dataset import SEDDataset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import torch.optim as optim
from torch.utils.data import  DataLoader
from metrics import F1Callback,mAPCallback,PANNsLoss
from catalyst.dl import SupervisedRunner,  CheckpointCallback,utils

def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

model_config = {
    "sample_rate": 16000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 5
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PANNsCNN14Att(**model_config)
model.to(device)


train_path = '/home/zonekey/project/audio_classification/train'
class_names_list1 = ['aloud_manual', 'clap', 'discuss_manual', 'noise', 'single_manual']
val_path = '/home/zonekey/project/audio_classification/clean_dataset/val_manual'
class_names_list2 = ['aloud', 'clap', 'discuss', 'noise', 'single']
wave_path_list1 = []
wave_path_list2 = []
label_list1 = []
label_list2 = []

for label_idx, class_name in enumerate(class_names_list1):
    class_dir = os.path.join(train_path, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith(".wav"):
            wave_path_list1.append(os.path.join(class_dir, filename))
            label_list1.append(label_idx)

label_counter1 = Counter(label_list1)
print("每類樣本數量:", label_counter1)

# 為每個樣本設置相對應的權重（樣本越少權重越高）
sample_weights1 = [1.0 / label_counter1[label] for label in label_list1]


sampler1 = WeightedRandomSampler(
    weights=sample_weights1,
    num_samples=len(sample_weights1),
    replacement=True
)

for label_idx, class_name in enumerate(class_names_list2):
    class_dir = os.path.join(val_path, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith(".wav"):
            wave_path_list2.append(os.path.join(class_dir, filename))
            label_list2.append(label_idx)

label_counter2 = Counter(label_list2)
print("每類樣本數量:", label_counter2)

# 為每個樣本設置相對應的權重（樣本越少權重越高）
sample_weights2 = [1.0 / label_counter2[label] for label in label_list2]
sampler2 = WeightedRandomSampler(
        weights=sample_weights2,
        num_samples=len(sample_weights2),
        replacement=True
    )



train_dataset = SEDDataset(wave_path_list=wave_path_list1, class_names_list=class_names_list1,augment=True)
val_dataset = SEDDataset(wave_path_list=wave_path_list2,  class_names_list=class_names_list2,augment=False )
train_loader = DataLoader(dataset=train_dataset, batch_size=5,sampler=sampler1)
val_loader = DataLoader(dataset=val_dataset, batch_size=5,sampler=sampler2 )

loaders = {
    "train": train_loader,
    "valid": val_loader }


checkpoint_path = "fold0/checkpoints/last.pth"  # ← 請改成你實際儲存的路徑
# 載入所有訓練狀態
checkpoint = utils.load_checkpoint(checkpoint_path)
# 恢復模型權重
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = PANNsLoss().to(device)
callbacks = [
    F1Callback(input_key="targets", output_key="logits", prefix="f1"),
    mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
    CheckpointCallback(save_n_best=0)
]
start_epoch = checkpoint.get("epoch", 0)
print('st:',start_epoch)
warnings.simplefilter("ignore")

runner = SupervisedRunner(
    device=device,
    input_key="waveform",
    input_target_key="targets",
import logging,time,os,torch,warnings
from typing import Optional, List
from contextlib import contextmanager
from model import PANNsCNN14Att
from dataset import SEDDataset
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import torch.optim as optim
from torch.utils.data import  DataLoader
from metrics import F1Callback,mAPCallback,PANNsLoss
from catalyst.dl import SupervisedRunner,  CheckpointCallback,utils

def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

model_config = {
    "sample_rate": 16000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 5
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PANNsCNN14Att(**model_config)
model.to(device)


train_path = '/home/zonekey/project/audio_classification/train'
class_names_list1 = ['aloud_manual', 'clap', 'discuss_manual', 'noise', 'single_manual']
val_path = '/home/zonekey/project/audio_classification/clean_dataset/val_manual'
class_names_list2 = ['aloud', 'clap', 'discuss', 'noise', 'single']
wave_path_list1 = []
wave_path_list2 = []
label_list1 = []
label_list2 = []

for label_idx, class_name in enumerate(class_names_list1):
    class_dir = os.path.join(train_path, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith(".wav"):
            wave_path_list1.append(os.path.join(class_dir, filename))
            label_list1.append(label_idx)

label_counter1 = Counter(label_list1)
print("每類樣本數量:", label_counter1)

# 為每個樣本設置相對應的權重（樣本越少權重越高）
sample_weights1 = [1.0 / label_counter1[label] for label in label_list1]


sampler1 = WeightedRandomSampler(
    weights=sample_weights1,
    num_samples=len(sample_weights1),
    replacement=True
)

for label_idx, class_name in enumerate(class_names_list2):
    class_dir = os.path.join(val_path, class_name)
    for filename in os.listdir(class_dir):
        if filename.endswith(".wav"):
            wave_path_list2.append(os.path.join(class_dir, filename))
            label_list2.append(label_idx)

label_counter2 = Counter(label_list2)
print("每類樣本數量:", label_counter2)

# 為每個樣本設置相對應的權重（樣本越少權重越高）
sample_weights2 = [1.0 / label_counter2[label] for label in label_list2]
sampler2 = WeightedRandomSampler(
        weights=sample_weights2,
        num_samples=len(sample_weights2),
        replacement=True
    )



train_dataset = SEDDataset(wave_path_list=wave_path_list1, class_names_list=class_names_list1,augment=True)
val_dataset = SEDDataset(wave_path_list=wave_path_list2,  class_names_list=class_names_list2,augment=False )
train_loader = DataLoader(dataset=train_dataset, batch_size=5,sampler=sampler1)
val_loader = DataLoader(dataset=val_dataset, batch_size=5,sampler=sampler2 )

loaders = {
    "train": train_loader,
    "valid": val_loader }


checkpoint_path = "fold0/checkpoints/last.pth"  # ← 請改成你實際儲存的路徑
# 載入所有訓練狀態
checkpoint = utils.load_checkpoint(checkpoint_path)
# 恢復模型權重
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
criterion = PANNsLoss().to(device)
callbacks = [
    F1Callback(input_key="targets", output_key="logits", prefix="f1"),
    mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
    CheckpointCallback(save_n_best=0)
]
start_epoch = checkpoint.get("epoch", 0)
print('st:',start_epoch)
warnings.simplefilter("ignore")

runner = SupervisedRunner(
    device=device,
    input_key="waveform",
    input_target_key="targets",sync_batchnorm=True
)

runner.train(
    model=model,
    criterion=criterion,
    loaders=loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5,
    verbose=True,
    logdir=f"fold0",
    callbacks=callbacks,
    main_metric="epoch_f1",
    minimize_metric=False)


)

runner.train(
    model=model,
    criterion=criterion,
    loaders=loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5,
    verbose=True,
    logdir=f"fold0",
    callbacks=callbacks,
    main_metric="epoch_f1",
    minimize_metric=False)





