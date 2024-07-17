from PIL import Image
from torchvision import transforms
import torch
import timm
import cv2

class Predict:
    def __init__(self):
        # Crear un modelo ConvNext
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Utilizando el dispositivo {self.device}.")
        self.num_classes = 4
        self.model = timm.create_model('convnext_large.fb_in22k_ft_in1k', pretrained=True, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load('./cnl4model-19.pt', map_location=torch.device('cpu')))
        
    def predict_img(self, image_path, isVideo = False):
        # Transformar la imagen de entrada mediante redimensionamiento y normalización
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

        # Cargar la imagen, preprocesarla y realizar predicciones
        if isVideo:
            img = Image.fromarray(image_path)
        else:
            img = Image.open(image_path)
            
        batch_t = torch.unsqueeze(transform(img), dim=0)
        batch_t = batch_t.to(self.device)
        self.model.eval()
        out = self.model(batch_t)

        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        # Devolver las 5 mejores predicciones ordenadas por las probabilidades más altas
        prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
        _, indices = torch.sort(out, descending=True)
        print(indices)
        return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
    
    def predict_video(self, video_path):
        
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        percentage = 0
        name = ''
        
        for frame in range(n_frames):
            ret, img = cap.read()
            if ret == False:
                break
            if frame % 100 == 0:
                print(f'Frame: {frame}')
                img_aux = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                labels = self.predict_img(img_aux, True)
                percentage = 0
                for l in labels:
                    if l[1] > percentage:
                        name = l[0]
                        percentage = l[1]
                if percentage > 95:
                    break
        
        percentage = percentage if (percentage > 90) else 0.0
        name = name if (percentage > 90) else "Desconocido"
        cap.release()
        print(labels)
        #print(percentages)
        
        return name, percentage