import os

import torch
import torch.nn as nn
import torchvision.models as models


def resnet18_fixed(num_classes):
    # Using a pre-trained ResNet model adapted for grayscale images
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer
    original_first_layer = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Initialize the new conv1 layer by averaging the weights of the original RGB channels
    with torch.no_grad():
        model.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
    # Adjust the fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(device)
    return model

def resnet18_dueling_dqn(num_classes, device):
    # Using a pre-trained ResNet model adapted for grayscale images
    model = models.resnet18(pretrained=True)
    # Modify the first convolutional layer to accept grayscale input
    original_first_layer = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)

    num_ftrs = model.fc.in_features
    # Replace the fully connected layer with a dueling architecture
    # State Value stream
    model.value_stream = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, 1)  # Scalar output representing V(s)
    )
    # Advantage stream
    model.advantage_stream = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)  # Outputs A(s, a) for each action
    )

    # Override the forward method to implement dueling architecture
    def forward(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        value = model.value_stream(x)
        advantages = model.advantage_stream(x)

        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    # Replace the default forward method with the dueling one
    model.forward = forward
    model = model.to(device)
    return model

# Initialize the original model and load the trained weights
def init_resnet18_fixed(num_classes, trained_weights_path):
    model = resnet18_fixed(num_classes)
    model.load_state_dict(torch.load(trained_weights_path))
    return model

# Initialize the dueling DQN model
def init_resnet18_dueling_dqn(num_classes):
    model = resnet18_dueling_dqn(num_classes, device)
    return model

# Function to copy weights from resnet18_fixed to resnet18_dueling_dqn
def transfer_weights(fixed_model, dueling_model):
    fixed_state_dict = fixed_model.state_dict()
    dueling_state_dict = dueling_model.state_dict()
    
    # Copy weights except for the final layers
    for name, param in fixed_state_dict.items():
        if name in dueling_state_dict and param.size() == dueling_state_dict[name].size():
            dueling_state_dict[name].copy_(param)
        else:
            print(f"Skipping {name} as it is not present or mismatched in the dueling model.")

    # Initialize the new layers (value and advantage streams)
    for name, param in dueling_model.named_parameters():
        if 'value_stream' in name or 'advantage_stream' in name:
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    return dueling_model


if __name__ == '__main__':
    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Usage
    # model = modify_and_load_model(original_model_path, num_classes)
    # get the absolute path of this python file folder
    current_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_path, 'pre-train_DQN.pth')

    num_classes = 6
    trained_weights_path = './EvaluationCNN/CNN.pth'

    fixed_model = init_resnet18_fixed(2, trained_weights_path)
    dueling_model = init_resnet18_dueling_dqn(num_classes)

    transfer_weights(fixed_model, dueling_model)
    # save the DQN model weights
    torch.save(dueling_model.state_dict(), save_path)