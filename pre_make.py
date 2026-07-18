import os
import cv2
import numpy as np
import glob

def batch_process_exposure_fusion_fixed():
    # 1. 定位基础路径（当前脚本与 train 同级）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(current_dir, "train")
    raw_data_dir = os.path.join(train_dir, "raw_data")
    trained_data_dir = os.path.join(train_dir, "trained_data")
    
    if not os.path.exists(raw_data_dir):
        print(f"【错误】未找到输入目录: {raw_data_dir}")
        return

    # 初始化 OpenCV Mertens 算法
    merge_mertens = cv2.createMergeMertens(contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.2)

    print("开始扫描并批量处理图像（材质全程_背景全程_照度 模式）...")
    print("-" * 70)

    # 2. 遍历材质目录 (aluminum, plated, zinc)
    for material in os.listdir(raw_data_dir):
        material_path = os.path.join(raw_data_dir, material)
        if not os.path.isdir(material_path):
            continue

        # 3. 遍历背景目录 (如 cloth_al, wood_al 等)
        for bg_dir in os.listdir(material_path):
            bg_path = os.path.join(material_path, bg_dir)
            if not os.path.isdir(bg_path):
                continue

            # 精准提取纯背景全程：从 "cloth_al" 或 "wood_al" 中切出 "cloth" 或 "wood"
            bg_name = bg_dir.split('_')[0]

            # 4. 遍历照度目录 (500, 600, 700 等)
            for lux_dir in os.listdir(bg_path):
                lux_path = os.path.join(bg_path, lux_dir)
                if not os.path.isdir(lux_path) or not lux_dir.isdigit():
                    continue
                
                # 精确查找当前照度目录下的所有 .bmp 文件
                search_path = os.path.join(lux_path, "*.[bB][mM][pP]")
                all_bmps = glob.glob(search_path)
                
                # 过滤出文件名纯为数字的曝光照片
                valid_paths = [p for p in all_bmps if os.path.splitext(os.path.basename(p))[0].isdigit()]
                
                if not valid_paths:
                    continue

                # 5. 读取并解码图像
                img_list = []
                for path in valid_paths:
                    img = cv2.imread(path)
                    if img is not None:
                        img_list.append(img)
                
                if not img_list:
                    print(f"【警告】目录 {lux_path} 下未成功解码任何有效图像，跳过。")
                    continue

                # 6. 执行 Mertens 曝光融合
                res_mertens = merge_mertens.process(img_list)
                res_8bit = np.clip(res_mertens * 255, 0, 255).astype(np.uint8)

                # 7. 拼装输出路径与规范化文件名
                # 输出子目录保持：train/trained_data/{material}
                output_sub_dir = os.path.join(trained_data_dir, material)
                os.makedirs(output_sub_dir, exist_ok=True)
                
                # 严格按照：材质全程_背景全程_照度.bmp 拼接
                output_filename = f"{material}_{bg_name}_{lux_dir}.bmp"
                output_file_path = os.path.join(output_sub_dir, output_filename)

                # 保存结果
                cv2.imwrite(output_file_path, res_8bit)
                print(f"【成功】已融合 [{material} -> 纯背景: {bg_name} -> {lux_dir}lx]")
                print(f"      └── 保存至: train/trained_data/{material}/{output_filename}")

    print("-" * 70)
    print("所有目录批量融合完成！")

if __name__ == "__main__":
    batch_process_exposure_fusion_fixed()