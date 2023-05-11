import glob
import pandas as pd
from pathlib import Path
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

DATA_ROOT = '/projectnb/rlvn/students/sfujita/carla-conditional-imitation-learning/logs/Combined/'
# DATA_ROOT = '/Users/shuhei/Desktop/carla-conditional-imitation-learning/logs/RHT/'

rgb_image_path = DATA_ROOT + 'images/*.jpg'
log_file_path = DATA_ROOT + 'log.csv'
command_file_path = DATA_ROOT + 'commands.csv'

class CILDataset(Dataset):
    def __init__(self):
        self.rgb_image_list = glob.glob(rgb_image_path)
        self.log_file = pd.read_csv(log_file_path)
        self.command_file = pd.read_csv(command_file_path)
        self.jitter = transforms.ColorJitter(brightness=.5, hue=.3)

    def __len__(self):
        return len(self.rgb_image_list)

    def __getitem__(self, index):
        """
        Load the RGB image, command and corresponding control outputs.
        Parameter:
            index (int): index of the data
        Return:
            image (torch.Tensor): RGB image 200x88x3
            commands (torch.Tensor): one-hot 8d vector of command (rht_, lht_) x (continue, left, straight, right)
            control_outputs (torch.Tensor): 2d vector (steer, throttle/brake)
        """
        image = read_image(self.rgb_image_list[index])
        image = self.jitter(image)
        frame_id = Path(self.rgb_image_list[index]).stem
        command = self._extract_command(frame_id)
        control_outputs = self._extract_control_outputs(frame_id)
        return image, command, control_outputs
    
    def _extract_command(self, frame_id):
        command_at_frame = self.command_file.loc[self.command_file['frame'] == frame_id]
        return command_at_frame[['rht_continue', 'rht_left', 'rht_straight', 'rht_right', 'lht_continue', 'lht_left', 'lht_straight', 'lht_right']].values[0]
    
    def _extract_control_outputs(self, frame_id):
        log_at_frame = self.log_file.loc[self.log_file['frame'] == frame_id]
        steer, throttle, brake = log_at_frame[['steer', 'throttle', 'brake']].values[0]
        return torch.tensor([steer, throttle + brake * -1])


def get_dataloader(batch_size, num_workers=4):
    return DataLoader(
        CILDataset(),
        batch_size=batch_size,
        num_workers=num_workers
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_loader = get_dataloader(32)
    image, command, control_outputs = next(iter(data_loader))
    print(f"Feature batch shape: {image.size()}")
    print(f"Labels batch shape: {command.size()}")
    print(f"Label: {command[0]}")
    print(f"Control outputs: {control_outputs[0]}")
    img = image[0].permute(1, 2, 0)
    plt.imshow(img, cmap="gray")
    plt.show()
