import os
import glob
import numpy as np
import SimpleITK as sitk
import pydicom
from skimage import morphology, measure, segmentation, filters
from scipy import ndimage as ndi


def load_dicom_series(dicom_dir):
    """加载DICOM系列并返回DICOM文件列表"""
    # 获取所有DICOM文件
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    
    if not dicom_files:
        raise ValueError(f"在 {dicom_dir} 中未找到DICOM文件")
    
    # 排序DICOM文件 - 使用更稳健的方法
    try:
        dicom_files = sorted(dicom_files, key=lambda x: float(pydicom.dcmread(x, stop_before_pixels=True).InstanceNumber))
    except:
        # 如果没有InstanceNumber，尝试使用文件名排序
        print("警告: 无法通过InstanceNumber排序，使用文件名排序")
        dicom_files = sorted(dicom_files)
    
    # 读取DICOM文件
    dicom_slices = []
    for file_path in dicom_files:
        dicom_slices.append(pydicom.dcmread(file_path))
    
    # 提取体积数据
    volume_data = np.stack([ds.pixel_array for ds in dicom_slices])
    
    # 应用Rescale Slope和Intercept (如果有)
    if hasattr(dicom_slices[0], 'RescaleSlope') and hasattr(dicom_slices[0], 'RescaleIntercept'):
        slope = dicom_slices[0].RescaleSlope
        intercept = dicom_slices[0].RescaleIntercept
        volume_data = volume_data * slope + intercept
        print(f"应用了像素值转换: 斜率={slope}, 截距={intercept}")
    
    # 获取体素间距
    try:
        spacing = (
            float(dicom_slices[0].SliceThickness),
            float(dicom_slices[0].PixelSpacing[0]),
            float(dicom_slices[0].PixelSpacing[1])
        )
    except:
        spacing = (1.0, 1.0, 1.0)
        print("警告: 未找到体素间距信息，使用默认值 [1,1,1]")
    
    # 获取窗宽窗位
    window_center = getattr(dicom_slices[0], 'WindowCenter', 0)
    window_width = getattr(dicom_slices[0], 'WindowWidth', 2000)
    
    # 如果是列表或序列取第一个值
    if hasattr(window_center, '__iter__'):
        window_center = window_center[0]
    if hasattr(window_width, '__iter__'):
        window_width = window_width[0]
    
    metadata = {
        'spacing': spacing,
        'window_center': window_center,
        'window_width': window_width
    }
    
    return dicom_slices, volume_data, metadata


def segment_with_otsu(volume_data):
    """使用Otsu自适应阈值的分割方法（优化版）"""
    depth, height, width = volume_data.shape
    mask_volume = np.zeros_like(volume_data, dtype=bool)
    
    print(f"开始使用Otsu方法分割 {depth} 个切片...")
    
    # 为了提高质量，我们首先分析整个体积的直方图
    # 找出全局合理的阈值范围
    flat_volume = volume_data.flatten()
    p1, p99 = np.percentile(flat_volume[flat_volume > flat_volume.min()], [1, 99])
    volume_range = p99 - p1
    
    for i in range(depth):
        # 获取当前切片
        slice_data = volume_data[i]
        
        # 使用Otsu方法自适应二值化
        # 改进的归一化策略
        norm_min = max(np.percentile(slice_data, 1), p1 - 0.1 * volume_range) 
        norm_max = min(np.percentile(slice_data, 99), p99 + 0.1 * volume_range)
        norm_image = np.clip((slice_data - norm_min) / (norm_max - norm_min + 1e-10), 0, 1)
        
        try:
            # 计算Otsu阈值
            otsu_thresh = filters.threshold_otsu(norm_image)
            # 应用阈值 - 肺部通常是暗区域
            binary = norm_image < otsu_thresh
        except:
            # 如果Otsu失败，使用固定阈值
            binary = norm_image < 0.4
        
        # 移除边界连接的区域
        cleared = segmentation.clear_border(binary)
        
        # 标记连通区域
        label_image = measure.label(cleared)
        regions = measure.regionprops(label_image)
        
        # 如果没有区域，我们尝试更宽松的阈值
        if not regions:
            binary = norm_image < 0.5  # 使用更宽松的阈值
            cleared = segmentation.clear_border(binary)
            label_image = measure.label(cleared)
            regions = measure.regionprops(label_image)
            
            # 如果仍然没有区域，继续下一个切片
            if not regions:
                continue
        
        # 选择面积最大的两个区域（这通常是左右肺）
        regions.sort(key=lambda x: x.area, reverse=True)
        max_regions = min(2, len(regions))
        
        # 创建掩码
        mask = np.zeros_like(binary)
        for j in range(max_regions):
            region_mask = label_image == regions[j].label
            mask |= region_mask
        
        # 增强形态学操作
        # 使用更大的结构元素进行闭操作，更好地连接肺部
        mask = morphology.binary_closing(mask, morphology.disk(7))
        # 填充空洞
        mask = ndi.binary_fill_holes(mask)
        # 移除小噪点
        mask = morphology.remove_small_objects(mask, min_size=150)
        
        # 保存掩码
        mask_volume[i] = mask
        
        # 每处理20个切片显示进度
        if (i + 1) % 20 == 0:
            print(f"已处理 {i+1}/{depth} 个切片")
    
    # 后处理整个掩码体积以确保Z方向的一致性
    for i in range(1, depth-1):
        # 使用前后切片来增强当前切片
        prev_mask = mask_volume[i-1]
        curr_mask = mask_volume[i]
        next_mask = mask_volume[i+1]
        
        # 如果当前切片与相邻切片差异很大，考虑调整
        if (np.sum(curr_mask) < 0.5 * np.sum(prev_mask)) or (np.sum(curr_mask) < 0.5 * np.sum(next_mask)):
            # 取相邻切片的并集作为基础
            combined = prev_mask | next_mask
            # 与当前切片取交集以保留共同区域
            adjusted = curr_mask | (combined & binary_dilation(curr_mask))
            mask_volume[i] = adjusted
    
    return mask_volume


def binary_dilation(binary_image, iterations=3):
    """对二值图像进行膨胀操作"""
    return ndi.binary_dilation(binary_image, 
                              structure=morphology.disk(3).astype(bool),
                              iterations=iterations)


def save_segmented_dicom(dicom_slices, mask_volume, output_dir):
    """将分割结果保存为DICOM文件，肺部外区域设置为最小可能值"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个DICOM文件
    for i, ds in enumerate(dicom_slices):
        if i >= len(mask_volume):
            print(f"警告: 掩码体积中缺少切片 {i}，跳过")
            continue
        
        try:
            # 复制原始数据集
            output_path = os.path.join(output_dir, f"segmented_{os.path.basename(ds.filename)}")
            ds_copy = ds.copy()
            
            # 获取原始像素数据
            orig_pixels = ds_copy.pixel_array.copy()
            
            # 确定数据类型和可能的最小值
            if ds_copy.pixel_array.dtype == np.uint16 or ds_copy.pixel_array.dtype == np.uint8:
                # 无符号类型：最小值为0
                min_possible_value = 0
            else:
                # 有符号类型：使用数据类型的最小可能值
                min_possible_value = np.iinfo(ds_copy.pixel_array.dtype).min
            
            # 目标HU值 (-1024, 标准空气值)
            target_hu = -1024
            
            # 计算对应存储值
            if hasattr(ds_copy, 'RescaleSlope') and hasattr(ds_copy, 'RescaleIntercept'):
                slope = float(ds_copy.RescaleSlope)
                intercept = float(ds_copy.RescaleIntercept)
                
                # 计算从HU值到存储值的转换: 存储值 = (HU值 - 截距) / 斜率
                # 比如 -1024 HU → ((-1024) - 0) / 1 = -1024 存储值 
                target_stored_value = int((target_hu - intercept) / slope)
                
                # 如果计算出的存储值小于数据类型允许的最小值，使用可能的最小值
                if target_stored_value < min_possible_value:
                    # 防止下溢
                    target_stored_value = min_possible_value
                    # 重新计算这个存储值对应的HU值，记录日志
                    actual_hu = target_stored_value * slope + intercept
                    print(f"警告: 切片 {i} - 无法表示 {target_hu} HU，使用最小可能值 {min_possible_value}，实际对应 {actual_hu:.1f} HU")
            else:
                # 没有缩放信息，直接使用原始数据类型最小值
                target_stored_value = min_possible_value
                print(f"警告: 切片 {i} - 无法确定转换关系，使用最小可能存储值 {min_possible_value}")
            
            # 应用分割掩码，直接将非肺部区域设置为目标值
            segmented_pixels = orig_pixels.copy()
            segmented_pixels[~mask_volume[i]] = target_stored_value
            
            # 确保像素类型与原始数据相同
            segmented_pixels = segmented_pixels.astype(orig_pixels.dtype)
            
            # 更新像素数据
            ds_copy.PixelData = segmented_pixels.tobytes()
            
            # 添加分割信息
            try:
                ds_copy.SeriesDescription = f"{ds.get('SeriesDescription', 'Series')}_Segmented"
            except:
                pass
            
            # 保存，明确指定转换语法
            ds_copy.is_little_endian = True
            ds_copy.is_implicit_VR = True
            ds_copy.save_as(output_path)
            
        except Exception as e:
            print(f"保存切片 {i} 时出错: {str(e)}")
            
            # 备用方法 - 使用SimpleITK
            try:
                print(f"尝试使用备用方法保存切片 {i}")
                # 创建分割后的像素数据
                orig_pixels = ds.pixel_array.copy()
                segmented_pixels = orig_pixels.copy()
                
                # 同样应用目标存储值
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    slope = float(ds.RescaleSlope)
                    intercept = float(ds.RescaleIntercept)
                    target_stored_value = int((target_hu - intercept) / slope)
                    
                    # 确保值在数据类型范围内
                    if ds.pixel_array.dtype == np.uint16 or ds.pixel_array.dtype == np.uint8:
                        target_stored_value = max(0, target_stored_value)
                    else:
                        type_info = np.iinfo(ds.pixel_array.dtype)
                        target_stored_value = max(type_info.min, min(type_info.max, target_stored_value))
                else:
                    # 没有缩放信息时使用数据类型最小值
                    if ds.pixel_array.dtype == np.uint16 or ds.pixel_array.dtype == np.uint8:
                        target_stored_value = 0
                    else:
                        target_stored_value = np.iinfo(ds.pixel_array.dtype).min
                
                segmented_pixels[~mask_volume[i]] = target_stored_value
                segmented_pixels = segmented_pixels.astype(orig_pixels.dtype)
                
                # 创建新的DICOM文件
                output_path = os.path.join(output_dir, f"segmented_{i:04d}.dcm")
                
                # 使用SimpleITK保存
                img = sitk.GetImageFromArray(segmented_pixels)
                
                # 复制必要的DICOM标签
                for tag in ['RescaleIntercept', 'RescaleSlope', 'WindowCenter', 'WindowWidth']:
                    if hasattr(ds, tag):
                        img.SetMetaData(tag, str(getattr(ds, tag)))
                
                sitk.WriteImage(img, output_path)
                print(f"使用备用方法保存切片 {i} 成功")
                
            except Exception as e:
                print(f"备用保存方法也失败: {str(e)}")
                # 保存为PNG以便检查
                try:
                    from PIL import Image
                    png_output_path = os.path.join(output_dir, f"segmented_{i:04d}.png")
                    
                    # 窗位窗宽归一化以便PNG显示
                    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                        wc = ds.WindowCenter
                        ww = ds.WindowWidth
                        if isinstance(wc, list): wc = wc[0]
                        if isinstance(ww, list): ww = ww[0]
                        
                        # 应用窗位窗宽
                        min_val = wc - ww/2
                        max_val = wc + ww/2
                        
                        # 归一化到0-255
                        display_pixels = np.clip(segmented_pixels, min_val, max_val)
                        display_pixels = ((display_pixels - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        # 简单归一化
                        display_pixels = segmented_pixels.astype(np.float32)
                        min_val = np.min(display_pixels)
                        max_val = np.max(display_pixels)
                        display_pixels = ((display_pixels - min_val) / (max_val - min_val + 1e-10) * 255).astype(np.uint8)
                    
                    Image.fromarray(display_pixels).save(png_output_path)
                    print(f"保存为PNG: {png_output_path}")
                except:
                    print(f"所有方法均失败，无法保存切片 {i}")
        
        # 每保存20个切片显示进度
        if (i + 1) % 20 == 0:
            print(f"已保存 {i+1}/{len(dicom_slices)} 个分割后的DICOM文件")
    
    print(f"分割结果已保存到: {output_dir}")

def analyze_segmentation_quality(mask_volume):
    """分析分割质量"""
    depth, height, width = mask_volume.shape
    total_voxels = depth * height * width
    lung_voxels = np.sum(mask_volume)
    lung_percentage = lung_voxels / total_voxels * 100
    
    print(f"分割结果分析:")
    print(f"  - 总体素数: {total_voxels}")
    print(f"  - 肺部体素数: {lung_voxels}")
    print(f"  - 肺部占比: {lung_percentage:.2f}%")
    
    # 每个切片的肺部体素数
    slice_coverage = [np.sum(mask_volume[i]) / (height * width) * 100 for i in range(depth)]
    
    # 找出肺部覆盖率最高的切片
    max_coverage_idx = np.argmax(slice_coverage)
    
    print(f"  - 肺部覆盖率最高的切片: #{max_coverage_idx} ({slice_coverage[max_coverage_idx]:.2f}%)")
    print(f"  - 平均切片肺部覆盖率: {np.mean(slice_coverage):.2f}%")
    
    return {
        'total_voxels': total_voxels,
        'lung_voxels': lung_voxels,
        'lung_percentage': lung_percentage,
        'slice_coverage': slice_coverage,
        'max_coverage_slice': max_coverage_idx
    }


def process_and_save_segmentation(input_dir, output_dir):
    """处理CT数据，分割肺部，并保存为DICOM格式"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载DICOM数据
    print(f"加载DICOM数据：{input_dir}")
    try:
        dicom_slices, volume_data, metadata = load_dicom_series(input_dir)
        print(f"加载了 {len(dicom_slices)} 个DICOM文件")
        print(f"体积大小: {volume_data.shape}")
        print(f"体素间距: {metadata['spacing']}")
        print(f"窗位/窗宽: {metadata['window_center']}/{metadata['window_width']}")
    except Exception as e:
        print(f"加载DICOM数据失败: {str(e)}")
        return
    
    # 分割肺部，使用优化的Otsu方法
    print("分割肺部区域...")
    mask_volume = segment_with_otsu(volume_data)
    
    # 分析分割质量
    quality_stats = analyze_segmentation_quality(mask_volume)
    
    # 如果分割完全失败（几乎没有肺部区域）
    if quality_stats['lung_voxels'] < 1000:
        print("警告: 分割几乎没有找到肺部区域，请检查分割结果")
        return
    
    # 保存分割后的DICOM
    print("保存分割结果为DICOM格式...")
    save_segmented_dicom(dicom_slices, mask_volume, output_dir)
    
    print("处理完成!")


if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = r"F:\dataset\pneumoconiosis\test\CT011228"
    output_dir = r"F:\dataset\pneumoconiosis\output"
    
    # 处理和保存
    process_and_save_segmentation(input_dir, output_dir)