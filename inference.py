import torch 
from RecNet.model import DigitRecognitionCNN
from torchvision import transforms
from PIL import Image
from utils.utils import save_model, load_model

model = DigitRecognitionCNN()
load_model(model, 'model.pth')
model.eval()

transform = transforms.Compose(
    transforms.Resize(28, 28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
)

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()
    
print(predict('path_to_image.png'))