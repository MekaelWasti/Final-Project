import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image
from io import BytesIO
import base64



device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # self.resnet = models.resnet50(pretrained=True)
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self,x):
        return self.resnet(x)
    

model = FacialSentimentAnalyser(len(intToLabelMap))
# checkpoint = "model_12-09-2023-4_46am.pth"
checkpoint = "resnet152-dec-10th-2_25am.pth"
model.load_state_dict(torch.load(checkpoint))
model.eval()
model.to(device)

def getSentiment(data, streamType):

    res = None
    predicted_probability = None

    if streamType == "image_stream":
        image = Image.open(BytesIO(data)).convert("RGB")
        imageTensor = transform(image)
        imageTensor = imageTensor.unsqueeze(0).to(device)
        
        # Model Inference
        logits = model(imageTensor)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        res = intToLabelMap[predictions.item()]
        predicted_probability = probabilities[0][predictions.item()].item() * 100  # To percentage

        print(res,predicted_probability)
    elif streamType == "video_stream":
        # header, encoded = data.split(',',1)
        # print(f'\n\n\n\n\nAH YEAH{data}\n\n\n\n')
        header, encoded = data.split(",", 1)
        decoded = base64.b64decode(encoded)
        # with open("temp_image.jpg", "wb") as f:
            # f.write(decoded)

        # print(f'\n\n\n\n\nAH YEAH{decoded}\n\n\n\n')
        try:
            image = Image.open(BytesIO(decoded))
            # image_to_tensor = transforms.ToTensor()
            # imageTensor1 = image_to_tensor(image)
            # image = np.array(image)

            # Torch Transform            
            imageTensor = transform(image)
            imageTensor = imageTensor.unsqueeze(0).to(device)
            
            # Model Inference
            logits = model(imageTensor)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)


            res = intToLabelMap[predictions.item()]
            predicted_probability = probabilities[0][predictions.item()].item() * 100  # To percentage
            print(res, predicted_probability)
        except Exception as e:
            print(f"Error loading image into PIL: {e}")
            res = ""
        # image = Image.open(BytesIO(decoded))

    # plt.imshow(image)
    # plt.show()

    # Process image
    

    return res, predicted_probability