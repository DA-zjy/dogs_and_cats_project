# predict.py

import torch
import pandas as pd
from tqdm import tqdm

# 从我们自己的模块中导入
import config
from model import ResNet34  # 确保这里是你从零训练时使用的模型
from dataset import get_test_loader # 我们只需要测试数据加载器

def predict_from_scratch():
    """使用从零训练的最佳模型进行预测，并生成提交文件"""
    print("开始使用从零训练的模型进行预测...")
    
    # 1. 定义你的最佳模型的路径
    MODEL_PATH = f"{config.CHECKPOINT_PATH}/dogs_and_cats_epoch_28.pth"
    
    # 2. 获取测试数据加载器 
    # 我们的 get_test_loader 会返回 (image, image_id)
    test_loader = get_test_loader()
    
    # 3. 实例化模型并移动到设备
    model = ResNet34(num_classes=2).to(config.DEVICE) # 确保类别数为2
    
    # 4. 加载你训练好的最佳模型权重
    print(f"正在加载模型: {config.CHECKPOINT_PATH}/dogs_and_cats_epoch_50.pth")
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # 5. 切换到评估模式
    model.eval()
    
    # 6. 存储预测结果
    all_image_ids = []
    all_dog_probs = []
    
    # 7. 开始预测循环
    with torch.no_grad():
        for images, image_ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(config.DEVICE)
            
            # --- 关键的预测步骤 ---
            # a. 获取模型原始输出 (logits)
            outputs = model(images)
            
            # b. 将 logits 转换为概率
            # torch.softmax 会输出每个类别的概率，比如 [[prob_cat, prob_dog], [prob_cat, prob_dog], ...]
            probs = torch.softmax(outputs, dim=1)
            
            # c. 提取“是狗的概率” (我们已经确认'dogs'的索引是1)
            prob_dog = probs[:, 1]
            # ---------------------
            
            # 将一个批次的结果添加到总列表中
            all_image_ids.extend(image_ids.cpu().numpy())
            all_dog_probs.extend(prob_dog.cpu().numpy())

    print("预测完成！")
    
    # 8. 生成提交文件
    print("正在生成提交文件...")
    submission = pd.DataFrame({
        "id": all_image_ids,
        "label": all_dog_probs
    })
    
    # 一个Kaggle小技巧：为了避免logloss惩罚，通常会将概率限制在一个很小的范围内，比如[0.005, 0.995]
    submission['label'] = submission['label'].clip(0.005, 0.995)
    
    submission_path = "./submission_from_scratch.csv"
    submission.to_csv(submission_path, index=False)
    
    print(f"提交文件已成功生成在: {submission_path}")
    print("现在你可以去Kaggle提交这个文件，获得你的基准分数了！")


if __name__ == '__main__':
    predict_from_scratch()