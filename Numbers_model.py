
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 4207
torch.manual_seed(SEED)
np.random.seed(SEED)

"""# Data Loading"""

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor()
)

def drop_zero(ds):
    idx = (ds.targets != 0).nonzero(as_tuple=True)[0]
    X = ds.data[idx].float().div(255).unsqueeze(1)
    y = ds.targets[idx] - 1
    return X, y

X_all, y_all = drop_zero(train_data)
X_test, y_test = drop_zero(test_data)

g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(X_all), generator=g)
n_val = int(len(X_all) * 10_000 / 60_000)
val_idx, tr_idx = perm[:n_val], perm[n_val:]
X_val, y_val = X_all[val_idx], y_all[val_idx]
X_tr,  y_tr  = X_all[tr_idx],  y_all[tr_idx]

loaders = {
    'train': DataLoader(TensorDataset(X_tr, y_tr), batch_size=100, shuffle=True),
    'val':   DataLoader(TensorDataset(X_val, y_val), batch_size=500),
    'test':  DataLoader(TensorDataset(X_test, y_test), batch_size=500),
}

"""# Model"""

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 9)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    train_loss = epoch_loss / len(loaders['train'])
    train_acc = 100. * correct / total
    print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
    return train_loss

def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loaders['test']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(loaders['test'])
    test_acc = 100. * correct / total
    print(f'Test set - Average Loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)\n')
    return test_loss, test_acc

for epoch in tqdm(range(1, 11), desc="Training Progress"):
  train(epoch)
test()

model.eval()

def get_probs(X):
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 500):
            out.append(F.softmax(model(X[i:i+500].to(device)), dim=1).cpu())
        return torch.cat(out).numpy()

np.save('probs_test.npy', get_probs(X_test))
np.save('labels_test.npy', (y_test + 1).numpy())
np.save('probs_val.npy', get_probs(X_val))
np.save('labels_val.npy', (y_val + 1).numpy())

torch.save(model.state_dict(), 'cnn.pt')

"""# Evaluation"""

pt, yt = np.load('probs_test.npy'), np.load('labels_test.npy')
pv, yv = np.load('probs_val.npy'),  np.load('labels_val.npy')

def ece(p, y, bins=15):
    conf, correct = p.max(1), (p.argmax(1) + 1 == y)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total

def reliability(p, y, title, bins=15):
    conf, correct = p.max(1), (p.argmax(1) + 1 == y)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.plot(xs, ys, 'o-'); plt.xlabel('confidence'); plt.ylabel('accuracy'); plt.title(title)

print('CNN test acc', (pt.argmax(1) + 1 == yt).mean(), 'ECE', ece(pt, yt))
print(confusion_matrix(yt, pt.argmax(1) + 1))

def scale(p, T):
    z = np.log(p + 1e-12) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

def nll(T, p, y):
    return -np.log(scale(p, T)[np.arange(len(y)), y - 1] + 1e-12).mean()

T = minimize_scalar(nll, bounds=(0.05, 10), args=(pv, yv), method='bounded').x
pt_cal = scale(pt, T)
print('T =', T, ' ECE after scaling:', ece(pt_cal, yt))
np.save('probs_test_cal.npy', pt_cal)
np.save('probs_val_cal.npy', scale(pv, T))

Xtr = X_tr.view(len(X_tr), -1).numpy(); ytr = (y_tr + 1).numpy()
Xte = X_test.view(len(X_test), -1).numpy()
lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
plr = lr.predict_proba(Xte)
print('LogReg test acc', (plr.argmax(1) + 1 == yt).mean(), 'ECE', ece(plr, yt))
print(confusion_matrix(yt, plr.argmax(1) + 1))

plt.figure(figsize=(12, 3.5))
plt.subplot(1, 3, 1); reliability(pt, yt, 'CNN')
plt.subplot(1, 3, 2); reliability(pt_cal, yt, 'CNN + temp. scaling')
plt.subplot(1, 3, 3); reliability(plr, yt, 'Logistic regression')
plt.tight_layout(); plt.show()
