from ultralytics import YOLO
import torch
import multiprocessing

def train_model():
    torch.cuda.empty_cache()
    
    model = YOLO(r'C:\Users\Aiswarya\Desktop\yol\yolov8n.pt')
    
    model.train(
        data=r'D:\datacam1\data.yaml',
        epochs=50,
        batch=8,
        imgsz=512,
        device=0,  
        val=True,
        workers=8,
        augment=True,
        patience=5,
        save=True,
        project=r'D:\results',
    )

if __name__ == '__main__':  
    multiprocessing.freeze_support()
    train_model()
