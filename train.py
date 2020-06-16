import os
import json
import argparse
import torch
import random
import torch.optim as optim
from model_vc import StyleEncoder
from model_vc import Generator
from dataset import AudiobookDataset
from dataset import train_collate
from dataset import test_collate
from utils.dsp import save_wav
import numpy as np
from hparams import hparams as hp

       
def save_checkpoint(device, model, optimizer, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_step{:06d}.pth".format(epoch))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer_state,
        "epoch": epoch
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def load_checkpoint(path, model, device, optimizer, reset_optimizer=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint['epoch'] 
    return epoch

def train(args, model, device, train_loader, optimizer, epoch, sigma=1.0):
    model.train()
    train_loss = 0

    for batch_idx, (m, p, e) in enumerate(train_loader):
        m = m.to(device)
        p = p.to(device)
        e = e.to(device)
        
        model.zero_grad()

        mel_outputs, mel_outputs_postnet, codes = model(m, e, p, e)

        m_rec = mel_outputs_postnet
        codes_rec = model(m_rec, e, None, None)

        L_recon = ((mel_outputs_postnet - m) ** 2).sum(dim=(1,2)).mean()
        L_recon0 = ((mel_outputs - m) ** 2).sum(dim=(1,2)).mean()
        L_content = torch.abs(codes - codes_rec).sum(dim=1).mean()

        loss = L_recon + L_recon0 + L_content

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(m)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(m), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))

def test(model, device, test_loader, checkpoint_dir, epoch, sigma=1.0):
    print("Using averaged model for evaluation")
    model.eval()
   
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (m, p, e) in enumerate(test_loader):
            m = m.to(device)
            p = p.to(device)
            e = e.to(device)
            
            mel_outputs, mel_outputs_postnet, codes = model(m, e, p, e)

            m_rec = mel_outputs_postnet
            codes_rec = model(m_rec, e, None, None)

            L_recon = ((mel_outputs_postnet - m) ** 2).sum(dim=(1,2)).mean()
            L_recon0 = ((mel_outputs - m) ** 2).sum(dim=(1,2)).mean()
            L_content = torch.abs(codes - codes_rec).sum(dim=1).mean()

            loss = L_recon + L_recon0 + L_content

            if batch_idx % 100 == 0:
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(m), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))
            test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run some neural net')
    parser.add_argument('-d', '--data', type=str, default='./data', help='dataset directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='The path to checkpoint')
    parser.add_argument('--epochs', type=int, default=600,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    data_path = args.data

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    torch.autograd.set_detect_anomaly(True)
    
    with open(os.path.join(data_path, 'train_data.json'), 'r') as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'test_data.json'), 'r') as f:
        test_data = json.load(f)

    train_loader = torch.utils.data.DataLoader(
        AudiobookDataset(train_data),
        collate_fn=train_collate,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        AudiobookDataset(test_data),
        collate_fn=test_collate,
        batch_size=1, shuffle=False, **kwargs)

    model = Generator(hp.dim_neck, hp.dim_emb, hp.dim_pitch, hp.dim_pre, hp.freq).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    current_epoch = 0
    if args.checkpoint:
        current_epoch = load_checkpoint(args.checkpoint, model, device, optimizer)
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(current_epoch + 1, args.epochs + 1):
        print(f'epoch {epoch}')
        train(args, model, device, train_loader, optimizer, epoch)

        if epoch % 10 == 0:
            test(model, device, test_loader, checkpoint_dir, epoch)
            save_checkpoint(device, model, optimizer, checkpoint_dir, epoch)
