import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
import io


intToLabelMap = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}

transform = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class FacialSentimentAnalyser(nn.Module):
    def __init__(self, num_classes):
        super(FacialSentimentAnalyser,self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self,x):
        return self.resnet(x)
    

model = FacialSentimentAnalyser(len(intToLabelMap))
checkpoint = "model_12-09-2023-4_46am.pth"
model.load_state_dict(torch.load(checkpoint))
model.eval()

def getSentiment(image):
    image = Image.open(io.BytesIO(image)).convert("RGB")
    # plt.imshow(image)
    # plt.show()

    # Process image
    imageTensor = transform(image)
    imageTensor = imageTensor.unsqueeze(0)
    logits = model(imageTensor)
    predictions = torch.argmax(logits, dim=1)

    res = intToLabelMap[predictions.item()]
    print(res)

    return res