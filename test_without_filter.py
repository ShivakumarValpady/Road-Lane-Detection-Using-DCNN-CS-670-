import time
import torch
import torch.nn as nn
from data import dataset
from torch.utils.data import DataLoader
from training import average, current_prog
from conv_lstm import Conv_LSTM
import training as tu
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from setup import device
from setup import Configs
import setup


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

configs = Configs()
print("Loading stored model")
model = Conv_LSTM(configs.hidden_dims, decoder_out_channels=2, lstm_nlayers=len(configs.hidden_dims),
                       vgg_decoder_config=configs.decoder_config)
tu.load_model_checkpoint(model, 'train-results/model.torch', inference=False, map_location=device)
model.to(device)
print("Model loaded")

tu_test_dataset = dataset(setup.test_dir, setup.test_sub,setup.test_labels, shuffle=False)
tu_test_dataloader = DataLoader(tu_test_dataset, batch_size=configs.test_batch, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.02, 1.02])).to(device)

test(tu_test_dataloader, model, criterion, log_every=1)