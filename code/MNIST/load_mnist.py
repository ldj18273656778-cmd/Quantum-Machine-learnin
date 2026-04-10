import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((4, 4)),  # 28x28 -> 4x4
    transforms.ToTensor(),       # [0,255] -> [0,1] 张量
])

# 2) 下载训练集/测试集
train_dataset = datasets.MNIST(
    root="./data",      # 保存目录
    train=True,         # 训练集
    download=True,      # 不存在就自动下载
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,        # 测试集
    download=True,
    transform=transform
)

# 3) 包装成 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("train size:", len(train_dataset))
print("test size:", len(test_dataset))
print("first batch shape:", next(iter(train_loader))[0].shape)
print("single image shape:", train_dataset[0][0].shape)

