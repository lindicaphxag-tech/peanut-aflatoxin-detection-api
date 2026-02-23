"""
坚果黄曲霉检测 - 后端API服务 (云端部署版)
"""
import os
import tempfile
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

app = Flask(__name__)
CORS(app)

# 模型路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model_resnet18.pth')

# 3分类类别名称
CLASS_NAMES = ['正常', '发霉不长毛', '发霉长毛']

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

_model = None

def get_model():
    """获取模型单例"""
    global _model
    if _model is None:
        print("正在加载模型...")
        _model = models.resnet18(weights=None)
        num_ftrs = _model.fc.in_features
        _model.fc = nn.Linear(num_ftrs, 3)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        _model.eval()
        print("✅ 模型加载成功")
    return _model

def preprocess_image(image_path):
    """图像预处理：背景去除"""
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB) / 255.0
    img_processed = (img_rgb * mask).astype(np.uint8)
    img_pil = Image.fromarray(img_processed)
    return transform(img_pil).unsqueeze(0)

def predict(image_path):
    """预测图片 - 3分类"""
    model = get_model()
    tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    moldy_prob = probs[1].item() + probs[2].item()

    return {
        'result': CLASS_NAMES[pred_idx],
        'confidence': round(probs[pred_idx].item() * 100, 1),
        'normal_prob': round(probs[0].item() * 100, 1),
        'moldy_light_prob': round(probs[1].item() * 100, 1),
        'moldy_heavy_prob': round(probs[2].item() * 100, 1),
        'moldy_prob': round(moldy_prob * 100, 1),
        'class_index': pred_idx
    }

@app.route('/api/detect', methods=['POST'])
def detect():
    """检测接口"""
    try:
        if 'image' in request.files:
            img = request.files['image']
            temp_path = os.path.join(tempfile.gettempdir(), 'detect_temp.jpg')
            img.save(temp_path)
            result = predict(temp_path)
            os.unlink(temp_path)
        elif request.json and 'base64' in request.json:
            img_data = base64.b64decode(request.json['base64'])
            temp_path = os.path.join(tempfile.gettempdir(), 'detect_temp.jpg')
            with open(temp_path, 'wb') as f:
                f.write(img_data)
            result = predict(temp_path)
            os.unlink(temp_path)
        else:
            return jsonify({'error': '请上传图片'}), 400

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'ResNet18-3class'})

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': '坚果黄曲霉检测API',
        'version': '1.0.0',
        'endpoints': {
            '/api/detect': 'POST - 上传图片检测',
            '/api/health': 'GET - 健康检查'
        }
    })

# 预加载模型
get_model()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

