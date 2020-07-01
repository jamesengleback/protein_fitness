import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np

class ProtDataset(Dataset):
    def __init__(self,path, cuda=False):
        self.df = pd.read_csv(path)
        self.cuda = cuda
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if not self.cuda:
            x = self.ohe(self.df.loc[idx,'Variants'])
            y = torch.tensor(self.df.loc[idx,'Fitness'])
            return x,y
        else:
            x = self.ohe(self.df.loc[idx,'Variants']).cuda()
            y = torch.tensor(self.df.loc[idx,'Fitness']).cuda()
            return x,y
    def ohe(self, seq):
        # one hot encoder - 2D
        aas = dict(zip(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
           'Q', 'R', 'S', 'T', 'V', 'W', 'Y'], range(20)))
        ohe = torch.zeros(4,20)
        for i,j in enumerate(seq):
            idx = aas[j]
            ohe[i,idx] = 1
        return ohe

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(80,8)
        self.layer2 = nn.Linear(8,1)
    def forward(self,x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x.squeeze(1) # same shape as y

def train(dataset, net, train_frac, lr, batch_size, epochs):
    # setup
    train_size = round(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    print(train_size, test_size)
    train_data, test_data = random_split(dataset,[train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=0)

    # train
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters(),lr=0.1)
    for epoch in range(epochs):
        for x,y in tqdm(train_loader):
            yh = net(x)
            loss = loss_fn(y,yh)
            loss.backward()
            opt.step()
            opt.zero_grad()
    # test
    net.eval()
    predictions = []
    ground_truths = []
    print('Test')
    for x,y in tqdm(test_loader):
        yh = net(x)
        predictions.append(yh)
        ground_truths.append(y)
    predictions = torch.cat(predictions).detach().numpy()
    ground_truths = torch.cat(ground_truths).detach().numpy()

    # score r
    r, p  = pearsonr(predictions,ground_truths)
    return r,p


def main():
    cuda = False
    dataset = ProtDataset('small-protein-fitness.csv', cuda)
    if cuda:
        net = Net().cuda()
    else:
        net = Net()
    # hyper params
    lr = 0.01
    batch_size = 8
    epochs = 5
    # loop thru train fracs
    rs,ps = [], []
    for train_frac in np.linspace(0.1,0.9,3):
        r,p = train(dataset,net, float(train_frac), lr, batch_size, epochs)
        rs.append(r), ps.append(p)

    df = pd.DataFrame([rs,ps])
    print(df)
if __name__ == '__main__':
    main()
