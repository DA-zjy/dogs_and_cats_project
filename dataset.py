import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split # 引入sklearn来做更灵活的拆分
from PIL import Image
import os
from config import BATCH_SIZE, TRAIN_PATH, TEST_PATH

# 我们可以创建一个通用的Dataset类，它接收图片路径列表和标签
class Train_Val_Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 根据索引获取图片路径和标签
        img_path = self.image_paths[index]
        label = self.labels[index]
        # 读取图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 为测试集创建一个专用的 Dataset 类
class Test_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 获取所有测试图片的文件名，并按文件名的数字大小进行排序
        # key=lambda x: int(x.split('.')[0]) 是为了确保 10.jpg 在 9.jpg 之后
        self.image_files = sorted(os.listdir(root_dir), key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 根据索引获取图片路径
        img_path = os.path.join(self.root_dir, self.image_files[index])
        
        # 从文件名中提取图片ID
        image_id = int(self.image_files[index].split('.')[0])
        
        # 读取图片
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 返回图片和它的ID
        return image, image_id
    
def get_train_val_loaders():
    """返回使用不同transform的训练和验证加载器"""
    
    # 1. 定义两种不同的transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. 先加载一次完整数据集，目的是为了获取所有图片的路径和标签
    full_dataset_info = datasets.ImageFolder(root=TRAIN_PATH)
    # .samples 属性是一个列表，每个元素是 (图片完整路径, 类别索引)
    image_paths = [item[0] for item in full_dataset_info.samples]
    labels = [item[1] for item in full_dataset_info.samples]

    # 3. 使用 sklearn 的 train_test_split 来拆分路径和标签（比random_split更灵活）
    # stratify=labels 能保证拆分后，训练集和验证集中的猫狗比例与原始数据集一致
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # 4. 为训练集和验证集分别创建Dataset实例，并传入各自的transform
    train_dataset = Train_Val_Dataset(train_paths, train_labels, transform=train_transform)
    val_dataset = Train_Val_Dataset(val_paths, val_labels, transform=val_transform)
    
    # 5. 创建DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

def get_test_loader():
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = Test_Dataset(TEST_PATH, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader

def get_dataloaders():
    train_loader, val_loader = get_train_val_loaders()
    test_loader = get_test_loader()
    return train_loader, val_loader, test_loader