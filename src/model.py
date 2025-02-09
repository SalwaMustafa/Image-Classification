from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader


image_transform = v2.Compose(
    [
        v2.Resize((299,299)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32 , scale = True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])





model = models.inception_v3(pretrained=False)
num_of_calsses = 6
model.fc = nn.Linear(model.fc.in_features, num_of_calsses)
print(model.load_state_dict(torch.load(r"D:\Image-Classification\inception_v3_weights.pth"  ,map_location=torch.device('cpu'))))
model.eval()




def generate_ans(file):

    image = Image.open(file)
    image_tensor = image_transform(image).unsqueeze(0)
    

    predicted_label = ""

    with torch.no_grad():
        out = model(image_tensor)
        _ , predicted_class = torch.max(out , 1)


        class_names = ['buildings','forest','glacier','mountain','sea','street']
        predicted_label = class_names[predicted_class]
        print(predicted_label)
        return predicted_label

    
    

