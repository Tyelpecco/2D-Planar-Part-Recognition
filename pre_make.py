import os
import cv2
import numpy as np
import glob

def process_numbered_bmp_group():
    # 获取当前 test.py 脚本所在的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"当前脚本运行目录已锁定: {current_dir}\n")

    # 精确查找该目录下的所有 .bmp 文件
    search_path = os.path.join(current_dir, "*.[bB][mM][pP]")
    all_bmps = glob.glob(search_path)
    
    # 过滤出文件名纯为数字的图片
    valid_pairs = []
    for path in all_bmps:
        filename_without_ext = os.path.splitext(os.path.basename(path))[0]
        if filename_without_ext.isdigit():
            valid_pairs.append((int(filename_without_ext), path))
            
    if not valid_pairs:
        print("【错误】在当前目录下没有找到任何以纯数字命名的 '.bmp' 图片！")
        print(f"请确认以下路径中是否存在图片: {current_dir}")
        return
        
    # 按照曝光度数字从小到大排序
    valid_pairs.sort(key=lambda x: x[0])
    
    print(f"检测到当前目录下待融合的 BMP 照片共 {len(valid_pairs)} 张：")
    img_list = []
    for exposure, path in valid_pairs:
        print(f" └── 成功载入 (曝光度: {exposure}): {os.path.basename(path)}")
        img = cv2.imread(path)
        if img is not None:
            img_list.append(img)
            
    if len(img_list) == 0:
        print("【错误】未成功解码任何图像，请确认 BMP 文件是否损坏。")
        return

    print("\n[处理中] 正在执行 Mertens 曝光融合算法（去除过曝与欠曝）...")
    
    # 初始化曝光融合算法
    merge_mertens = cv2.createMergeMertens(contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.2)
    res_mertens = merge_mertens.process(img_list)
    
    # 转换回工业标准的 8 位无符号整数 (0~255)
    res_8bit = np.clip(res_mertens * 255, 0, 255).astype(np.uint8)
    
    # 输出无损 BMP 结果，同样保存在该目录下
    output_filename = os.path.join(current_dir, "merged_result.bmp")
    cv2.imwrite(output_filename, res_8bit)
    
    print("-" * 50)
    print(f"【成功】工业工件图像融合完成！")
    print(f"结果已保存至: {output_filename}")

if __name__ == "__main__":
    process_numbered_bmp_group()