import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 1. โหลดคลาสและโครงสร้างโมเดลที่เทรนไว้
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('durian_model.pth', map_location=device)
class_names = checkpoint['class_names']

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 2. แปลงรูปภาพให้ตรงกับตอนเทรน
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# คู่มือคำแนะนำเบื้องต้นสำหรับแต่ละโรค
recommendations = {
    'Anthracnose': 'โรคแอนแทรคโนส: แนะนำให้ตัดแต่งกิ่งที่เป็นโรคออกเผาทำลาย และพ่นด้วยสารกำจัดเชื้อรา เช่น คาร์เบนดาซิม หรือไดฟีโนโคนาโซล',
    'Canker': 'โรคแคงเกอร์: เกิดจากแบคทีเรีย ควรตัดแต่งกิ่งและพ่นสารคอปเปอร์ไฮดรอกไซด์เพื่อควบคุมการแพร่ระบาด',
    'Leaf_Algal': 'โรคจุดสาหร่าย: เกิดจากความชื้นสูงและแดดไม่พอ ควรตัดแต่งทรงพุ่มให้โปร่งและพ่นสารคอปเปอร์ออกซิคลอไรด์',
    'Healthy': 'ใบทุเรียนปกติสมบูรณ์: พืชมีความแข็งแรงดี รักษาสภาพการให้น้ำและปุ๋ยตามรอบปกติได้เลยครับ'
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์รูปภาพ'}), 400
    
    file = request.files['file']
    try:
        image = Image.open(file.stream).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted = torch.max(probabilities, 0)
            
        predicted_class = class_names[predicted.item()]
        conf_score = confidence.item() * 100
        
        advice = recommendations.get(predicted_class, 'ดูแลรักษาตามอาการและปรึกษาผู้เชี่ยวชาญ')

        return jsonify({
            'disease': predicted_class,
            'confidence': round(conf_score, 2),
            'advice': advice
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("--- เริ่มต้นเซิร์ฟเวอร์ AI ทำนายโรคทุเรียนที่พอร์ต 5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=True)