import glob
import torch
import torchvision
import matplotlib.pyplot as plt

DIRECTORY_PATH = '/Users/shuhei/Desktop/carla-conditional-imitation-learning/'
MODEL_PATH = DIRECTORY_PATH + 'agent.pth'

def get_command_tensor(self, command='rht_continue'):
    if command == 'rht_continue':
        return torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    elif command == 'rht_left':
        return torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    elif command == 'rht_straight':
        return torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])
    elif command == 'rht_right':
        return torch.tensor([0, 0, 0, 1, 0, 0, 0, 0])
    elif command == 'lht_continue':
        return torch.tensor([0, 0, 0, 0, 1, 0, 0, 0])
    elif command == 'lht_left':
        return torch.tensor([0, 0, 0, 0, 0, 1, 0, 0])
    elif command == 'lht_straight':
        return torch.tensor([0, 0, 0, 0, 0, 0, 1, 0])
    else: # command == 'lht_right'
        return torch.tensor([0, 0, 0, 0, 0, 0, 0, 1])


path_to_images = glob.glob(DIRECTORY_PATH + 'logs/RHT/images/*.jpg')
print('Number of images: {}'.format(len(path_to_images)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

COUNT = 30
for image_path in path_to_images[:min(COUNT, len(path_to_images))]:
    image = torchvision.io.read_image(image_path)
    image_tensor = image.to(device).unsqueeze(0).float()
    command_tensor = get_command_tensor('straight').to(device).unsqueeze(0).float()
    with torch.no_grad():
        output = model(image_tensor, command_tensor)
        print('inference output', output)
    # plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    # plt.show()
