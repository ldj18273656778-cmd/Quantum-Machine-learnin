import torchvision
import torchvision.transforms as transforms
import numpy as np

def binarize(x):
    return (x > 0.5).float()

transform = transforms.Compose([
    transforms.Resize((10, 10)),
    transforms.ToTensor(),
    transforms.Lambda(binarize)
])

dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
X = []
y = []

for img, label in dataset:
    X.append(img.numpy())   # (1,10,10)
    y.append(label)

X = np.array(X)  # (N,1,10,10)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)