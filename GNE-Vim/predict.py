import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from models.iqa_model import VimIQAModel
from utils.process_image import ToTensor

class SimplePredictor:
    def __init__(self, weight_path, image_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_path = weight_path
        self.image_size = image_size
        self.create_model()
        self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            ToTensor()
        ])

    def create_model(self):
        self.model = VimIQAModel(checkpoint=self.weight_path)  
        self.model.to(self.device)

    def load_model(self):
        checkpoint = torch.load(self.weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, image_paths):
        for image_path in image_paths:
            d_img = Image.open(image_path).convert('RGB')
            d_img_tensor = self.transform(d_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, pred_score = self.model(d_img_tensor, d_img_tensor)  
                print(f'Image: {image_path}, Predicted Score: {pred_score.item()}')

def main():
    parser = argparse.ArgumentParser(description='Predict image quality scores.')
    parser.add_argument('weight_path', type=str, help='Path to the model weight file.')
    parser.add_argument('image_size', type=int, help='Size to resize images to.')
    parser.add_argument('image_paths', nargs='+', help='Paths to the images to predict.')

    args = parser.parse_args()
    
    predictor = SimplePredictor(args.weight_path, args.image_size)
    predictor.predict(args.image_paths)

if __name__ == '__main__':
    main()
