import cv2
from analyzer_api import WorkpieceAnalyzerAPI

# 初始化分析器
analyzer = WorkpieceAnalyzerAPI("C:/Users/Tyelpecco/Desktop/test/workpiece_svm_model.pkl")

# 1. 先用 cv2.imread 读取图片矩阵
img_matrix = cv2.imread("C:/Users/Tyelpecco/Desktop/test/1.bmp")

# 2. 确保图片读取成功后再传给 predict
if img_matrix is not None:
    material, score = analyzer.predict(img_matrix)
else:
    print("【错误】图片读取失败，请检查路径是否正确！")
print(f"识别结果：该工件是 【{material}】，置信度：{score*100:.2f}%")

# 3. 根据材质去调用不同的机械臂抓取策略
if material == "plated":
    print("触发电镀件特殊吸盘/夹爪位姿解算...")
    # 执行电镀件的位姿解算与分拣...