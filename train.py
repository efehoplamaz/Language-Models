from typing import Iterable
import torch
from tqdm import tqdm
from dataset import LanguageModelDataset

def training_step(train_dataloader: Iterable, model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer, epoch: int,
                  device: torch.device, batch_size: int):

    model.train()
    pbar = tqdm(enumerate(train_dataloader, 0), unit=' images', unit_scale=batch_size,
                total=len(train_dataloader), smoothing=0, disable= False)

    running_loss = 0.0
    log_freq = 10
   
    for (i, data) in pbar:
        optimizer.zero_grad()
        logits, loss = model(data['input'].to(device), data['target'].to(device))
        running_loss += loss.detach()
        loss.backward()  
        optimizer.step()

        if i % log_freq == 0:
            pbar.set_description('Train [ Epoch: {}, Loss: {:.4f}, Average Loss: {}]'.format(epoch, float(loss), float(running_loss) / (i + 1)))

    average_epoch_loss = float(running_loss/(i + 1))

    print('Train [ Epoch: {}, Average Loss: {}]'.format(epoch, average_epoch_loss))

def train(model:torch.nn.Module, corpora_path: str,
          block_size: int, learning_rate: float,
          batch_size: int, epoch: int):

    ds = LanguageModelDataset(corpora_path=corpora_path, block_size=block_size)
    language_sequence_dataloader = torch.utils.data.DataLoader(ds, collate_fn=None, shuffle=True, num_workers=1, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch_num in range(epoch):
        training_step(train_dataloader=language_sequence_dataloader,
                      model=model,
                      optimizer=optimizer,
                      epoch=epoch_num,
                      device= "cuda" if torch.cuda.is_available() else "cpu",
                      batch_size=batch_size)
    