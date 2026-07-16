import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern

class WorkpieceAnalyzerAPI:
    def __init__(self, model_path="workpiece_svm_model.pkl"):
        """初始化时自动加载训练好的模型"""
        self.model = joblib.load(model_path)
        self.class_labels = {0: "aluminum", 1: "zinc", 2: "plated"}
        
    def _extract_features(self, img):
        """内部私有函数：提取单张图像的特征（逻辑与训练时严格一致）"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_features = [
            np.mean(hsv[..., 0]), np.std(hsv[..., 0]),
            np.mean(hsv[..., 1]), np.std(hsv[..., 1]),
            np.mean(hsv[..., 2]), np.std(hsv[..., 2])
        ]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return np.array(color_features + hist.tolist()).reshape(1, -1)

    def predict(self, fused_image):
        """
        供外部调用的公开 API 接口
        :param fused_image: 传入的图像矩阵 (numpy array)，即经多曝光融合后的单组图像
        :return: (str 零件材质种类, float 置信度百分比)
        """
        if fused_image is None:
            return "unknown", 0.0
            
        # 1. 提取特征
        features = self._extract_features(fused_image)
        
        # 2. 预测类别
        pred_code = self.model.predict(features)[0]
        material_name = self.class_labels.get(pred_code, "unknown")
        
        # 3. 计算该预测的置信度（概率）
        probs = self.model.predict_proba(features)[0]
        confidence = probs[pred_code]
        
        return material_name, float(confidence)