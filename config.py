# config.py

import torch

# --- 训练配置 ---
DEVICE = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 64 #Batch如何选择？
LEARNING_RATE = 0.001 # 0.1

# --- 模型保存配置 ---
CHECKPOINT_PATH = "./checkpoints"

#数据集路径
TRAIN_PATH = "./data/train_sorted"
TEST_PATH = "./data/test"