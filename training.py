import time
import torch

from model import CILModel
from dataset import get_dataloader

save_path = './agent.pth'
N_COMMANDS = 4
BATCH_SIZE = 120
EPOCHS = 100

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    infer_action = CILModel().to(device)
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=0.0002)
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()
    train_loader = get_dataloader(BATCH_SIZE)
    for epoch in range(EPOCHS):
        total_loss = 0
        for _, batch in enumerate(train_loader):
            batch = [b.to(device) for b in batch]
            batch_image, batch_commands, batch_control_outputs = batch
            batch_out = infer_action(batch_image.float(), batch_commands)
            loss = criterion(batch_out, batch_control_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (EPOCHS - 1 - epoch)
        print('Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs' % (
            epoch + 1, total_loss, time_left))
    torch.save(infer_action, save_path)


if __name__ == "__main__":
    train()
