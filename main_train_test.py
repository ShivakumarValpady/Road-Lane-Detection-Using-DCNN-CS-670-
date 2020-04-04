import time
import torch
import torch.nn as nn
from data import dataset
from torch.utils.data import DataLoader
from training import average
from training import current_prog
from conv_lstm import Conv_LSTM
import training as tu
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from setup import device
from setup import Configs
import setup

####################train

def train(train_loader:DataLoader, model:Conv_LSTM, criterion, optimizer, epoch, log_every=1):

    batch_time = average('BatchTime', ':6.3f')
    data_time = average('Data', ':6.3f')
    losses = average('Loss', ':.4e')
    acc = average('Accuracy', ':6.4f')
    f1 = average('F1 score', ':6.4f')
    prec = average('Precision', ':6.4f')
    rec = average('Recall', ':6.4f')
    progress = current_prog(
        len(train_loader),
        [batch_time, data_time, losses, acc, f1, prec, rec],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for batch_no, (batch_imgs, ground_truth_img) in enumerate(train_loader):
        data_time.update(time.time() - end)
                
        for j in range(len(batch_imgs)):
            le = batch_imgs[j].shape[0]
            test_samples = []
            for le1 in range(le):
                aa1 = batch_imgs[j]
                a1 = aa1[le1]
                a1 = ndimage.maximum_filter(a1, size=6)
                a1 = ndimage.gaussian_filter(a1,sigma = 1.5)
                a1 = torch.from_numpy(a1).float().to(device)
                a1 = a1.unsqueeze(0)      
                test_samples.append(a1)
            batch_imgs[j] = torch.cat([i for i in test_samples])

        batch_imgs = [t.to(device) for t in batch_imgs]

        ground_truth_img = ground_truth_img.long().to(device)
        ground_truth_img = ground_truth_img.squeeze(1)
        output = model(batch_imgs)
        loss = criterion(output, ground_truth_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.detach()
        losses.update(loss.item(), ground_truth_img.size(0))

        ground_truth_img = ground_truth_img.float()
        accuracy = pixel_accuracy(output, ground_truth_img)
        acc.update(accuracy, ground_truth_img.size(0))
        f, (p, r) = f1_score(output, ground_truth_img)

        f1.update(f)
        prec.update(p)
        rec.update(r)
        
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_no % log_every == 0:
            print("Output min", output.min().item(), "Output (softmax-ed) sum:", (output > 0.).float().sum().item(), "Output max:", torch.max(output).item())
            print("Targets sum:", ground_truth_img.sum())
            print("Base acc:{} - base prec: {}- base recall: {}- base f1: {}".
                  format(pixel_accuracy(output, ground_truth_img), p, r, f))
            progress.display(batch_no)

    return losses.avg, acc.avg, f1.avg

################test
def test(val_loader, model, criterion, log_every=1):
    batch_time = average('Time', ':6.3f')
    losses = average('Loss', ':.4e')
    acc = average('Acc', ':6.4f')
    f1 = average('F1', ':6.4f')
    prec = average('Prec', ':6.4f')
    rec = average('Recall', ':6.4f')
    progress = current_prog(
        len(val_loader),
        [batch_time, losses, acc, f1, prec, rec],
        prefix='Test: ')

    #model.eval()
    model.train()

    correct = 0
    error = 0
    precision = 0.
    recall = 0.
    with torch.no_grad():
        end = time.time()
        for batch_no, (test_samples, gt_test) in enumerate(val_loader):
            
            for j in range(len(test_samples)):
                le = test_samples[j].shape[0]
                samples1 = []
                for le1 in range(le):
                    aa1 = test_samples[j]
                    a1 = aa1[le1]
                    a1 = ndimage.maximum_filter(a1, size=6)
                    a1 = ndimage.gaussian_filter(a1,sigma = 1.5)
                    a1 = torch.from_numpy(a1).float().to(device)
                    a1 = a1.unsqueeze(0)      
                    samples1.append(a1)
                test_samples[j] = torch.cat([i for i in samples1])
                
            test_samples = [t.to(device) for t in test_samples]
            
            
            gt_test = gt_test.squeeze(1).long().to(device)

            output = model(test_samples)

            loss = criterion(output, gt_test)
            losses.update(loss.item(), gt_test.size(0))

            f, (p, r) = f1_score(output, gt_test.float())
            f1.update(f)
            prec.update(p)
            rec.update(r)
 
            acc.update(pixel_accuracy(output, gt_test), gt_test.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_no % log_every == 0:
                progress.display(batch_no)

        return acc.avg


def pixel_accuracy(prediction:torch.Tensor, target:torch.Tensor):
    out = (prediction[:, 1, :, :] > 0.).float()
    return (out == target).float().mean().item()

def f1_score(output, target, epsilon=1e-7):
    probas = (output[:, 1, :, :] > 0.).float()
    TP = (probas * target).sum(dim=1)
    precision = TP / (probas.sum(dim=1) + epsilon)
    recall = TP / (target.sum(dim=1) + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return f1.mean().item(), (precision.mean().item(), recall.mean().item())

def adjust_learning_rate(optimizer, epoch, init_lr):
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

configs = Configs()
epochs = configs.epochs
init_lr = configs.init_lr
batch_size = configs.batch_size
workers = configs.workers
momentum = configs.momentum
weight_decay = configs.weight_decay
hidden_dims = configs.hidden_dims
decoder_config = configs.decoder_config

#################training
tu_tr_dataset = dataset(setup.training_dir,setup.training_sub, setup.training_labels, shuffle=False)#, shuffle_seed=9)

tu_train_dataloader = DataLoader(tu_tr_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

model = Conv_LSTM(hidden_dims, decoder_out_channels=2, lstm_nlayers=len(hidden_dims), vgg_decoder_config=decoder_config)

if configs.load_model:
    tu.load_model_checkpoint(model, 'train-results/model.torch', inference=False, map_location=device)

model.to(device)

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).to(device)

optimizer = torch.optim.Adam(model.parameters(), init_lr)

losses_1 = []

for epoch in range(epochs):
    loss_val, a, f = train(tu_train_dataloader, model, criterion, optimizer, epoch, log_every=16)
    losses_1.append(loss_val)
    tu.save_model_checkpoint(model, 'train-results/model11.torch', epoch=epoch)
    
    plt.plot(losses_1)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title("Plot for Training loss")
    plt.savefig("train-results/training_loss.png")
    
#################### testing
 
configs = Configs()
print("Loading model")
model = Conv_LSTM(configs.hidden_dims, decoder_out_channels=2, lstm_nlayers=len(configs.hidden_dims),
                       vgg_decoder_config=configs.decoder_config)
tu.load_model_checkpoint(model, 'train-results/model.torch', inference=False, map_location=device)
model.to(device)
print("Loading Completed Succesfully")

tu_test_dataset = dataset(setup.test_dir, setup.test_sub,setup.test_labels, shuffle=False)#, shuffle_seed=9)

tu_test_dataloader = DataLoader(tu_test_dataset, batch_size=configs.test_batch, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).to(device)

test(tu_test_dataloader, model, criterion, log_every=1)