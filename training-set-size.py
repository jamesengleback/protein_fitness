import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import argparse

from ax.service.managed_loop import optimize

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
    print('Train')
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
    predictions = torch.cat(predictions).detach().cpu().numpy()
    ground_truths = torch.cat(ground_truths).detach().cpu().numpy()

    # score r
    r, p  = pearsonr(predictions,ground_truths)
    return r



def plot_(df):
     plt.plot(df['train_frac'], df['R'])
     plt.ylabel('r')
     plt.xlabel('train frac')
     plt.ylim(0,1)
     plt.xlim(0,1)
     plt.title('training sizes vs accuracy')
     plt.grid()
     plt.savefig('training.png')

def main(args):

    dataset = ProtDataset(args.data, args.cuda)
    # loop thru train fracs
    epochs = 5
    rs,ps, fracs = [], [], []
    for train_frac in np.linspace(0.01, 0.99,50):
        if args.cuda:
            net = Net().cuda()
        else:
            net = Net()
        r = train(dataset,net, train_frac, lr, batch_size, epochs)
        rs.append(r), ps.append(p), fracs.append(train_frac)

    df = pd.DataFrame([fracs,rs,ps], index = ['train_frac','R','P']).T
    df.to_csv('scores.csv')
    plot_(df)
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data')
    parser.add_argument('-c','--cuda', action = 'store_true')
    args = parser.parse_args()
    main(args)
