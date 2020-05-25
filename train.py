import os
import json
import argparse
import torch
import torch.optim as optim
from model_vc import StyleEncoder
from model_vc import Generator
from dataset import AudiobookDataset
from dataset import train_collate
from dataset import test_collate
from utils.dsp import save_wav
import numpy as np

def save_checkpoint(device, model, style_enc, optimizer, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_step{:06d}.pth".format(epoch))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "model": model.state_dict(),
        "style": style_enc.state_dict(),
        "optimizer": optimizer_state,
        "epoch": epoch
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def train(args, model, style_enc, device, train_loader, optimizer, epoch, sigma=1.0):
    model.train()
    tram_loss = 0

    for batch_idx, (m, _, _) in enumerate(train_loader):
        m = m.to(device)
        
        model.zero_grad()

        m = m.transpose(2,1)
        emb = style_enc(m)
        mel_outputs, mel_outputs_postnet, codes = model(m, emb, emb)

        m_rec = mel_outputs_postnet
        emb_rec = style_enc(m_rec)
        codes_rec = model(m_rec, emb_rec, None)

        L_recon = ((mel_outputs_postnet - m) ** 2).mean()
        L_recon0 = ((mel_outputs - m) ** 2).mean()
        L_content = torch.abs(codes - codes_rec).mean()

        loss = L_recon + L_recon0 + L_content

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(m), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    
    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))

def test(model, device, test_loader, checkpoint_dir, epoch, sigma=1.0):
    model.eval()
   
    test_loss = 0

    with torch.no_grad():
        for m, _, fname in test_loader:
            m = m.transpose(2,1)
            emb = style_enc(m)
            mel_outputs, mel_outputs_postnet, codes = model(m, emb, emb)

            m_rec = mel_outputs_postnet
            emb_rec = style_enc(m_rec)
            codes_rec = model(m_rec, emb_rec, None)

            L_recon = ((mel_outputs_postnet - m) ** 2).mean()
            L_recon0 = ((mel_outputs - m) ** 2).mean()
            L_content = torch.abs(codes - codes_rec).mean()

            loss = L_recon + L_recon0 + L_content
            test_loss += loss.item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run some neural net')
    parser.add_argument('-d', '--data', type=str, default='./data', help='dataset directory')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--batch-size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    data_path = args.data

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    torch.autograd.set_detect_anomaly(True)
    
    with open(os.path.join(data_path, 'train.json'), 'r') as f:
        train_index = json.load(f)

    with open(os.path.join(data_path, 'test.json'), 'r') as f:
        test_index = json.load(f)

    train_loader = torch.utils.data.DataLoader(
        AudiobookDataset(train_index),
        collate_fn=train_collate,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        AudiobookDataset(test_index),
        collate_fn=test_collate,
        batch_size=1, shuffle=False, **kwargs)

    dim_neck = 32
    dim_emb = 256
    dim_pre = 512
    freq = 32

    style_enc = StyleEncoder(dim_emb).to(device)
    model = Generator(dim_neck, dim_emb, dim_pre, freq).to(device)

    optimizer = optim.Adam([p for p in model.parameters()] + [p for p in style_enc.parameters()], lr=args.lr)
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f'epoch {epoch}')
        train(args, model, style_enc, device, train_loader, optimizer, epoch)

        if epoch % 10 == 0:
            test(model, device, test_loader, checkpoint_dir, epoch)
            save_checkpoint(device, model, style_enc, optimizer, checkpoint_dir, epoch)
