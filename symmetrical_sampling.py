'''
对称采样，k张切片，均匀分布
'''

import os
import shutil
import numpy as np
from pathlib import Path


def symmetrical_sampling(patient_dir, output_dir, k=64):
    """
    对一个病人的CT切片进行对称采样，均匀选取k张切片
    
    参数:
        patient_dir: 包含病人CT切片的目录路径
        output_dir: 采样后切片的保存目录
        k: 需要采样的切片数量，默认为64
    
    返回:
        采样的切片文件路径列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有切片文件
    slice_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
        slice_files.extend(list(Path(patient_dir).glob(f'*{ext}')))
    
    # 按文件名排序，假设文件名中包含切片序号信息
    slice_files.sort()
    
    total_slices = len(slice_files)
    
    if total_slices == 0:
        print(f"警告: 在{patient_dir}中未找到任何图片文件")
        return []
    
    if total_slices <= k:
        # 如果切片总数小于等于k，则全部选取
        selected_indices = list(range(total_slices))
    else:
        # 计算均匀间隔
        step = (total_slices - 1) / (k - 1) if k > 1 else 0
        
        # 生成对称均匀分布的索引
        selected_indices = [int(round(i * step)) for i in range(k)]
    
    # 获取选中的切片路径
    selected_slices = [slice_files[i] for i in selected_indices]
    
    # 复制选中的切片到输出目录
    copied_slices = []
    for slice_file in selected_slices:
        dest_file = os.path.join(output_dir, slice_file.name)
        shutil.copy2(slice_file, dest_file)
        copied_slices.append(dest_file)
    
    return copied_slices


def batch_symmetrical_sampling(input_base_dir, output_base_dir, k=64):
    """
    对多个病人的CT切片进行批量对称采样
    
    参数:
        input_base_dir: 包含多个病人文件夹的基础目录
        output_base_dir: 采样结果的输出基础目录
        k: 每个病人需要采样的切片数量，默认为64
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 获取所有病人文件夹
    patient_dirs = [d for d in os.listdir(input_base_dir) 
                   if os.path.isdir(os.path.join(input_base_dir, d))]
    
    from tqdm import tqdm
    for patient_dir in tqdm(patient_dirs):
        patient_input_path = os.path.join(input_base_dir, patient_dir)
        patient_output_path = os.path.join(output_base_dir, patient_dir)
        
        print(f"处理病人: {patient_dir}")
        selected_slices = symmetrical_sampling(patient_input_path, patient_output_path, k)
        print(f"已为病人{patient_dir}选择{len(selected_slices)}张切片")


if __name__ == "__main__":
    import argparse
    
    input_root = "G:\CC2\origin_data\gt0"
    output_root = "G:\CC2\origin_data\gt0_s16"
    k = 16

    for i in range(0, 3):
        input = os.path.join(input_root, str(i))
        # output = os.path.join(output_root, str(i))
        batch_symmetrical_sampling(input, output_root, k)






