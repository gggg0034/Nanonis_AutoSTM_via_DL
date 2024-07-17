import torch
import torch.nn as nn
import torch.nn.functional
import torchvision.models as models
from PIL import Image
from torchvision import transforms


def load_model(model_path):
    # Assume using the modified ResNet18 model
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image_quality(image_data, model_path):
    # Load the trained model
    model = load_model(model_path)
    model.eval()  # Set the model to evaluation mode
    
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Load and transform the image
    # if image_data is a file path
    if isinstance(image_data, str):
        image = Image.open(image_data).convert('L')  # Convert to grayscale
    else:
        image = Image.fromarray(image_data).convert('L')
    # if image_data is a Tensor
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(image)
        
        _, predicted = torch.max(outputs.data, 1)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        # Get the idx = 1 probability
        probability = outputs.data[0][1].item()        
        # return predicted.item()
        return probability



if __name__ =='__main__':
    # Example usage:
    model_path = './EvaluationCNN/CNN.pth'
    image_path = 'C:/Users/73594/Desktop/my python/SPM-AI-nanonis/STM-demo/mustard_AI_nanonis/DQN/memory/equalize/Scan_data_back12-56-41.png'
    result = predict_image_quality(image_path, model_path)
    print('This is a {} picture.'.format('good' if result >= 0.5 else 'bad'))
