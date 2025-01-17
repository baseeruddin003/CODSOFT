import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 128)  

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
siamese_model = SiameseNetwork()
siamese_model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_faces(frame):
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def recognize_faces(frame, faces, reference_embedding):
   
    results = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        try:
            face_tensor = transform(face_img).unsqueeze(0)  
            face_embedding, _ = siamese_model(face_tensor, face_tensor)

            # Calculating (cosine similarity)
            similarity = torch.nn.functional.cosine_similarity(face_embedding, reference_embedding).item()
            results.append({'box': (x, y, w, h), 'similarity': similarity})
        except Exception as e:
            print(f"Recognition error: {e}")
            results.append({'box': (x, y, w, h), 'similarity': None})

    return results

def main(video_source=0, reference_image_path=None):
   
    cap = cv2.VideoCapture(video_source)

    reference_embedding = None
    if reference_image_path:
        reference_image = cv2.imread(reference_image_path)
        reference_image = transform(reference_image).unsqueeze(0)
        reference_embedding, _ = siamese_model(reference_image, reference_image)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from video source.")
            break

        faces = detect_faces(frame)
        results = recognize_faces(frame, faces, reference_embedding) if reference_embedding is not None else []

       
        for result in results:
            (x, y, w, h) = result['box']
            similarity = result['similarity']

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            if similarity is not None:
                info = f"Similarity: {similarity:.2f}"
                cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Face Detection and Recognition', frame)
        #press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(reference_image_path="test.png")
