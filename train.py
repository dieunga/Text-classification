from ultralytics import YOLO

# Load pretrained model
model = YOLO("/content/drive/MyDrive/sample_intern1/title_crop.pt") 

# Train the model
results = model.train(data="/content/drive/MyDrive/sample_intern1/dataset.yaml", epochs=100, imgsz=640)