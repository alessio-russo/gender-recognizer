import cv2 as cv
import torch
from torchvision import transforms

model = torch.load('model.pt')
height, width = 224, 224

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((height, width))
])

camera = cv.VideoCapture(0)
for i in range(8):
    result, image = camera.read()

if result:
    data = transform(image)
    output = model(data.unsqueeze(0))
    label = torch.nn.functional.sigmoid(output)
    gender = 'female' if label >= 0.5 else 'male'

    cv.rectangle(image, (int(image.shape[0]/2) + 100, 570), (int(image.shape[0]/2) + 550, 620), (0,0,0), thickness=cv.FILLED)
    cv.putText(image, f"You are a {gender}!!", (int(image.shape[0]/2) + 180, 605), cv.FONT_HERSHEY_TRIPLEX,
               1.0, (255, 255,255) , 2)
    cv.imshow('hello!', image)

    cv.waitKey(0)

    camera.release()