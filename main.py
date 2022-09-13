import argparse
import os

import torch

import Utils
import Data.Market as mk
import Data.DGDataset as dataset
import Network.DGNet as DGNet
import trainer as tr
train_output = './Data/classified_market/train/'
test_output = './Data/classified_market/test/'
query_output = './Data/classified_market/query/'

parser = argparse.ArgumentParser(description='DGNet.')
parser.add_argument('--seed', metavar='seed', type=int, default=42, help='Number of seed')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch', metavar='batch', type=int, default=8, help='Number of batch size')
parser.add_argument('--epoch', metavar='epoch', type=int, default=100, help='Number of epoch')
parser.add_argument('--data_dir', metavar='data_dir', type=str, default='./Data/Market/', help='path of data')
args = parser.parse_args()

learning_rate = 1e-2
batch_size = 8
num_workers = 0
weight_decay = 0
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available!  Training on GPU ...')
else:
    device = torch.device('cpu')
    print('CUDA is not available.  Training on CPU ...')

Utils.seed_everything(args.seed)

if not os.path.exists(train_output):
    print('Preparing Market...')
    mk.prepare_train_test(args.data_dir + "bounding_box_train/", train_output)
    mk.prepare_train_test(args.data_dir + "bounding_box_test/", test_output)
    mk.prepare_train_test(args.data_dir + "query/", query_output)

market_dataset = dataset.DG_Dataset(train_output)


train_loader = torch.utils.data.DataLoader(market_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True, drop_last = True)

trainer = tr.Trainer()
optimizer = torch.optim.Adam(trainer.parameters(), lr = learning_rate, weight_decay = weight_decay)

for i in range(args.epoch):
    trainer.train()
    for xj_gt, xi_gt, xj, xi, xt in enumerate(train_loader):
        optimizer.zero_grad()
        xj = xj.to(device)
        xi = xi.to(device)
        xt = xt.to(device)
        Xj_gen, Xi_gen, Xt_gen, Xj_s, Xi_a, Xi_s, Xt_a, Xi_x, Xt_x, teacher_Xi_a = trainer.forward(xj, xi, xt)

        loss = trainer.update(xj, xi, xt, xi_gt, xj_gt, Xj_gen, Xi_gen, Xt_gen, Xj_s, Xi_a, Xi_s, Xt_a, Xi_x, Xt_x, teacher_Xi_a)

        loss.backward()
        optimizer.step()

    print("Epoch : " + str(i) + ", loss : " + str(loss.cpu().detach().numpy()))
