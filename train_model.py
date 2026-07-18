import os
import cv2
import numpy as np
import joblib  # 用于保存和加载模型
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# =====================================================================
# 1. 核心特征提取函数（必须与未来 API 接口中的提取逻辑完全一致）
# =====================================================================
def extract_workpiece_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    # --- 特征 1：HSV 色彩统计特征 ---
    # 目的：利用电镀件会微弱倒影麻布/木桌颜色的特性，提取色调和饱和度差异
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv[..., 0]), np.std(hsv[..., 0])
    s_mean, s_std = np.mean(hsv[..., 1]), np.std(hsv[..., 1])
    v_mean, v_std = np.mean(hsv[..., 2]), np.std(hsv[..., 2])
    color_features = [h_mean, h_std, s_mean, s_std, v_mean, v_std]
    
    # --- 特征 2：LBP (局部二值模式) 纹理特征 ---
    # 目的：区分铝的拉丝划痕、锌的压铸颗粒感以及电镀的极致光滑度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # LBP 参数：radius为半径，n_points为邻域像素点数
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # 计算 LBP 的直方图作为纹理特征向量（uniform模式下有 n_points + 2 个bin）
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    texture_features = hist.tolist()
    
    # 将颜色特征与纹理特征拼接成一个总特征向量
    global_features = color_features + texture_features
    return global_features

# =====================================================================
# 2. 加载数据集
# =====================================================================
def load_dataset(data_dir):
    X = []  # 特征向量列表
    y = []  # 标签列表
    
    # 类别映射字典
    class_map = {"aluminum": 0, "zinc": 1, "plated": 2}
    
    for class_name, class_label in class_map.items():
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.exists(class_folder):
            continue
            
        print(f"正在读取 {class_name} 类别的图片...")
        for filename in os.listdir(class_folder):
            if filename.lower().endswith(('.bmp', '.jpg', '.png')):
                img_path = os.path.join(class_folder, filename)
                features = extract_workpiece_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(class_label)
                    
    return np.array(X), np.array(y)

# =====================================================================
# 3. 主训练流程
# =====================================================================
if __name__ == "__main__":
    DATA_DIR = "./train/trained_data"  # 你的数据存放目录
    
    if not os.path.exists(DATA_DIR):
        print(f"【错误】找不到 {DATA_DIR} 文件夹，请先创建并放入分类数据！")
        exit()
        
    # 载入数据与特征提取
    X, y = load_dataset(DATA_DIR)
    print(f"\n数据集加载完成！样本总数: {len(X)}，每个样本特征维度: {X.shape[1]}")
    
    if len(X) == 0:
        print("【错误】未提取到任何有效样本，请检查图片格式。")
        exit()
        
    # 划分训练集和测试集（80% 训练，20% 验证）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("正在训练 SVM 分类模型...")
    # 初始化支持向量机分类器（C为惩罚系数，kernel='linear'或'rbf'均可，工业场景linear很稳定）
    model = SVC(C=1.0, kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*20 + " 训练测试报告 " + "="*20)
    print(f"验证集准确率 (Accuracy): {accuracy * 100:.2f}%")
    print("\n详细分类指标:")
    target_names = ["aluminum (铝)", "zinc (锌)", "plated (电镀)"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 保存模型文件，用于未来的 API 调用
    model_filename = "workpiece_svm_model.pkl"
    joblib.dump(model, model_filename)
    print(f"【成功】模型已成功序列化并保存为: {model_filename}")