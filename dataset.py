import glob
import pandas as pd
from pathlib import Path
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

rgb_image_path = '/projectnb/rlvn/students/sfujita/final-project/logs/530896036299243230-1681240837.092009/images/*.jpg'
log_file_path = '/projectnb/rlvn/students/sfujita/final-project/logs/530896036299243230-1681240837.092009/logs.csv'
command_file_path = '/projectnb/rlvn/students/sfujita/final-project/logs/530896036299243230-1681240837.092009/commands.csv'

class CILDataset(Dataset):
    def __init__(self):
        self.rgb_image_list = glob.glob(rgb_image_path)
        self.log_file = pd.read_csv(log_file_path)
        self.command_file = pd.read_csv(command_file_path)

    def __len__(self):
        return len(self.rgb_image_list)

    def __getitem__(self, index):
        """
        Load the RGB image, command and corresponding control outputs.
        Parameter:
            index (int): index of the data
        Return:
            image (torch.Tensor): RGB image 200x88x3
            commands (torch.Tensor): one-hot 4d vector of command (continue, left, straight, right)
            control_outputs (torch.Tensor): 3d vector (steer, throttle, brake)
        """
        image = read_image(self.rgb_image_list[index])
        frame_id = Path(self.rgb_image_list[index]).stem
        commands = self.command_file.loc[self.command_file['frame'] == int(frame_id)]
        commands = commands[['continue', 'left', 'straight', 'right']].values[0]
        measurements = self.log_file.loc[self.log_file['frame'] == int(frame_id)]
        control_outputs = measurements[['steer', 'throttle', 'brake']].values[0]
        return image, commands, control_outputs


def get_dataloader(batch_size, num_workers=4):
    return DataLoader(
        CILDataset(),
        batch_size=batch_size,
        num_workers=num_workers
    )

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_loader = get_dataloader(32)
    image, command, control_outputs = next(iter(data_loader))
    print(f"Feature batch shape: {image.size()}")
    print(f"Labels batch shape: {command.size()}")
    img = image[0].permute(1, 2, 0)
    label = command[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    print(f"Control outputs: {control_outputs[0]}")
