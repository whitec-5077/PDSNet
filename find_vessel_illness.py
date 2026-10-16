import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from skimage.filters import frangi
from scipy import ndimage
import glob
from tqdm import tqdm
import time
import shutil

def dicom_to_hu(ds):
    """
    Convert stored DICOM pixel values to HU when CT rescale metadata are available.
    Missing metadata default to the identity transform.
    """
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return image * slope + intercept

def hu_to_stored_pixels(ds, hu_image):
    """
    Convert HU values back to stored DICOM pixel values before writing PixelData.
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    stored = (hu_image.astype(np.float32) - intercept) / slope

    try:
        dtype = ds.pixel_array.dtype
    except Exception:
        dtype = np.int16

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        stored = np.clip(np.rint(stored), info.min, info.max).astype(dtype)
    else:
        stored = stored.astype(dtype)
    return stored

def dicom_sort_key(file_path):
    """
    Sort DICOM slices by spatial metadata when available.
    Consistent z-order is important for 3D Frangi filtering and 3D connectivity.
    """
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
    except pydicom.errors.InvalidDicomError:
        try:
            ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
        except Exception:
            return (4, os.path.basename(file_path))
    except Exception:
        return (4, os.path.basename(file_path))

    if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
        try:
            return (0, float(ds.ImagePositionPatient[2]), os.path.basename(file_path))
        except Exception:
            pass

    if hasattr(ds, "InstanceNumber"):
        try:
            return (1, float(ds.InstanceNumber), os.path.basename(file_path))
        except Exception:
            pass

    if hasattr(ds, "SliceLocation"):
        try:
            return (2, float(ds.SliceLocation), os.path.basename(file_path))
        except Exception:
            pass

    return (3, os.path.basename(file_path))

def load_dicom_series(dicom_dir):
    """
    加载DICOM序列
    """
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files:
        print(f"警告: 在 {dicom_dir} 中未找到DICOM文件")
        return None, None
    
    # 过滤有效的DICOM文件
    valid_dicom_files = []
    for file in dicom_files:
        try:
            # 尝试读取DICOM文件
            test_dicom = pydicom.dcmread(file, stop_before_pixels=True)
            valid_dicom_files.append(file)
        except pydicom.errors.InvalidDicomError:
            try:
                # 尝试强制读取
                test_dicom = pydicom.dcmread(file, force=True, stop_before_pixels=True)
                valid_dicom_files.append(file)
                # print(f"警告: 强制读取可能非标准的DICOM文件: {file}")
            except Exception as e:
                print(f"跳过无效文件: {file}, 错误: {e}")
                continue
    
    if not valid_dicom_files:
        print(f"警告: 在 {dicom_dir} 中未找到有效的DICOM文件")
        return None, None

    valid_dicom_files.sort(key=dicom_sort_key)
    
    print(f"找到 {len(valid_dicom_files)} 个有效DICOM文件")
        
    # 读取第一个有效文件以获取维度
    try:
        ref_dicom = pydicom.dcmread(valid_dicom_files[0])
    except pydicom.errors.InvalidDicomError:
        ref_dicom = pydicom.dcmread(valid_dicom_files[0], force=True)
    
    # 估计CT的背景值（阈值最小值），使用HU空间以保持跨扫描一致性
    background_value = np.min(dicom_to_hu(ref_dicom))
    print(f"检测到背景值: {background_value}")
    
    # 创建3D体积
    volume = np.zeros((len(valid_dicom_files), ref_dicom.pixel_array.shape[0], ref_dicom.pixel_array.shape[1]), dtype=np.float32)
    
    print("正在加载DICOM序列...")
    successful_loads = 0
    for i, file in enumerate(tqdm(valid_dicom_files)):
        try:
            ds = pydicom.dcmread(file)
        except pydicom.errors.InvalidDicomError:
            try:
                ds = pydicom.dcmread(file, force=True)
            except Exception as e:
                print(f"无法读取文件 {file}: {e}")
                continue
        
        volume[i, :, :] = dicom_to_hu(ds)
        successful_loads += 1
    
    if successful_loads == 0:
        print("错误: 没有成功加载任何DICOM文件")
        return None, None
    
    # 如果部分文件加载失败，调整体积大小
    if successful_loads < len(valid_dicom_files):
        volume = volume[:successful_loads]
        print(f"警告: 只成功加载了 {successful_loads}/{len(valid_dicom_files)} 个文件")
    
    return volume, background_value

def normalize_lung_volume(lung_ct, lung_mask, lower_percentile=1.0, upper_percentile=99.5):
    """
    Normalize the lung field to [0, 1] before 3D Frangi filtering.
    Robust percentiles reduce the influence of isolated calcifications or noise.
    """
    normalized = np.zeros_like(lung_ct, dtype=np.float32)
    lung_values = lung_ct[lung_mask].astype(np.float32)
    if lung_values.size == 0:
        return normalized

    lower, upper = np.percentile(lung_values, [lower_percentile, upper_percentile])
    if upper <= lower:
        lower = float(lung_values.min())
        upper = float(lung_values.max())
    normalized[lung_mask] = np.clip(
        (lung_ct[lung_mask].astype(np.float32) - lower) / (upper - lower + 1e-8),
        0.0,
        1.0
    )
    return normalized

def build_high_confidence_vessel_tree(
    enhanced_vessels,
    lung_mask,
    seed_threshold,
    connectivity_structure,
    min_seed_size=100,
    min_seed_fraction=1e-4
):
    """
    Build a conservative vessel-tree prior from strong, continuous 3D Frangi
    responses. Small disconnected high-response regions are intentionally not
    included, so they can remain eligible as opacity-like candidates.
    """
    seed_mask = (enhanced_vessels > seed_threshold) & lung_mask
    seed_mask = ndimage.binary_closing(seed_mask, structure=connectivity_structure)
    seed_mask &= lung_mask

    labeled_seeds, num_seeds = ndimage.label(seed_mask, structure=connectivity_structure)
    if num_seeds == 0:
        return np.zeros_like(seed_mask, dtype=bool), labeled_seeds, num_seeds

    seed_sizes = np.bincount(labeled_seeds.ravel(), minlength=num_seeds + 1)
    lung_voxels = int(lung_mask.sum())
    tree_min_size = max(int(min_seed_size), int(lung_voxels * min_seed_fraction))
    keep_labels = np.where(seed_sizes >= tree_min_size)[0]
    keep_labels = keep_labels[keep_labels != 0]

    if keep_labels.size == 0:
        largest_label = int(np.argmax(seed_sizes[1:]) + 1)
        if seed_sizes[largest_label] >= min_seed_size:
            keep_labels = np.array([largest_label], dtype=np.int32)

    if keep_labels.size == 0:
        return np.zeros_like(seed_mask, dtype=bool), labeled_seeds, num_seeds

    vessel_tree = np.isin(labeled_seeds, keep_labels)
    return vessel_tree, labeled_seeds, num_seeds

def segment_vessels_frangi(
    lung_ct,
    background_value,
    sigma_range=(1, 3),
    sigma_step=1,
    threshold_factor=0.6,
    candidate_percentile=70,
    seed_percentile=90,
    min_component_size=5,
    min_seed_size=100,
    min_seed_fraction=1e-4,
    connectivity_radius=1
):
    """
    使用3D Frangi滤波和连通血管树先验进行保守血管抑制。
    
    参数:
    - lung_ct: 肺部CT体积
    - background_value: 背景值
    - sigma_range: Frangi滤波器中sigma的范围
    - sigma_step: sigma增量步长
    - threshold_factor: 候选高响应区域阈值因子
    - candidate_percentile: 候选高响应区域的分位数阈值
    - seed_percentile: 高置信血管树种子的分位数阈值
    - min_component_size: 删除前保留的最小3D连通域大小
    - min_seed_size: 高置信血管树种子的最小连通域大小
    - min_seed_fraction: 高置信血管树种子相对肺体素数的最小比例
    - connectivity_radius: 判断候选组件是否连接到血管树的空间容差
    
    返回:
    - vessel_mask: 血管掩码
    - enhanced_vessels: 3D Frangi增强后的图像
    """
    start_time = time.time()
    
    lung_mask = (lung_ct != background_value)
    if not np.any(lung_mask):
        print("警告：未检测到肺部区域")
        return np.zeros_like(lung_ct, dtype=np.uint8), np.zeros_like(lung_ct, dtype=np.float32)

    normalized_volume = normalize_lung_volume(lung_ct, lung_mask)
    sigmas = np.arange(sigma_range[0], sigma_range[1] + sigma_step, sigma_step)

    print("正在应用3D Frangi滤波器增强管状结构...")
    enhanced_vessels = frangi(
        normalized_volume,
        sigmas=sigmas,
        black_ridges=False,
        beta=0.8,
        gamma=30.0
    )
    enhanced_vessels = np.nan_to_num(enhanced_vessels, nan=0.0, posinf=0.0, neginf=0.0)
    enhanced_vessels = enhanced_vessels.astype(np.float32)
    enhanced_vessels *= lung_mask.astype(np.float32)

    frangi_time = time.time() - start_time
    print(f"3D Frangi滤波完成，耗时: {frangi_time:.2f} 秒")
    
    start_time = time.time()

    non_zero_values = enhanced_vessels[(enhanced_vessels > 0) & lung_mask]
    if non_zero_values.size == 0:
        print("警告：未检测到增强的管状结构")
        return np.zeros_like(lung_ct, dtype=np.uint8), enhanced_vessels

    candidate_threshold = np.percentile(non_zero_values, candidate_percentile) * threshold_factor
    seed_threshold = np.percentile(non_zero_values, seed_percentile)
    seed_threshold = max(seed_threshold, candidate_threshold + 1e-8)

    print("正在进行3D连通域分析和血管树连通性判定...")
    structure_26 = ndimage.generate_binary_structure(3, 3)

    candidate_mask = (enhanced_vessels > candidate_threshold) & lung_mask
    candidate_mask = ndimage.binary_closing(candidate_mask, structure=structure_26)
    candidate_mask &= lung_mask

    labeled_candidates, num_candidates = ndimage.label(candidate_mask, structure=structure_26)
    if num_candidates == 0:
        print("警告：未检测到高响应候选组件")
        return np.zeros_like(lung_ct, dtype=np.uint8), enhanced_vessels

    vessel_tree, labeled_seeds, num_seeds = build_high_confidence_vessel_tree(
        enhanced_vessels=enhanced_vessels,
        lung_mask=lung_mask,
        seed_threshold=seed_threshold,
        connectivity_structure=structure_26,
        min_seed_size=min_seed_size,
        min_seed_fraction=min_seed_fraction
    )

    if connectivity_radius > 0:
        connected_tree = ndimage.binary_dilation(
            vessel_tree,
            structure=structure_26,
            iterations=int(connectivity_radius)
        )
    else:
        connected_tree = vessel_tree
    connected_tree &= lung_mask

    candidate_sizes = np.bincount(labeled_candidates.ravel(), minlength=num_candidates + 1)
    connected_labels = np.unique(labeled_candidates[connected_tree & candidate_mask])
    connected_labels = connected_labels[connected_labels != 0]
    connected_labels = np.array(
        [label for label in connected_labels if candidate_sizes[label] >= min_component_size],
        dtype=np.int32
    )

    if connected_labels.size == 0:
        vessel_mask = np.zeros_like(lung_ct, dtype=np.uint8)
    else:
        vessel_mask = np.isin(labeled_candidates, connected_labels).astype(np.uint8)
        vessel_mask &= lung_mask.astype(np.uint8)

    preserved_components = num_candidates - connected_labels.size
    
    post_time = time.time() - start_time
    print(f"候选阈值: {candidate_threshold:.6f}, 血管树种子阈值: {seed_threshold:.6f}")
    print(f"3D候选组件数: {num_candidates}, 高置信种子组件数: {num_seeds}")
    print(f"连接到血管树并被抑制的组件数: {connected_labels.size}")
    print(f"保留的非血管连通高响应组件数: {preserved_components}")
    print(f"3D连通性后处理完成，耗时: {post_time:.2f} 秒")
    
    return vessel_mask, enhanced_vessels

def save_as_dicom(original_dicom_dir, output_dicom_dir, processed_ct, case_name, series_description="Processed CT"):
    """
    将处理后的CT保存为DICOM格式
    """
    os.makedirs(output_dicom_dir, exist_ok=True)
    
    dicom_files = glob.glob(os.path.join(original_dicom_dir, "*.dcm"))
    if not dicom_files:
        print(f"警告: 在 {original_dicom_dir} 中未找到DICOM文件")
        return
    
    # 过滤有效的DICOM文件
    valid_dicom_files = []
    for file in dicom_files:
        try:
            # 尝试读取DICOM文件
            test_dicom = pydicom.dcmread(file, stop_before_pixels=True)
            valid_dicom_files.append(file)
        except pydicom.errors.InvalidDicomError:
            try:
                # 尝试强制读取
                test_dicom = pydicom.dcmread(file, force=True, stop_before_pixels=True)
                valid_dicom_files.append(file)
            except Exception as e:
                print(f"跳过无效文件: {file}, 错误: {e}")
                continue
    
    if not valid_dicom_files:
        print(f"警告: 在 {original_dicom_dir} 中未找到有效的DICOM文件")
        return
    
    valid_dicom_files.sort(key=dicom_sort_key)
    
    print(f"正在保存{series_description} - 病例: {case_name}")
    successful_saves = 0
    for i, file in enumerate(tqdm(valid_dicom_files)):
        if i < processed_ct.shape[0]:
            try:
                ds = pydicom.dcmread(file)
            except pydicom.errors.InvalidDicomError:
                try:
                    ds = pydicom.dcmread(file, force=True)
                except Exception as e:
                    print(f"无法读取文件 {file}: {e}")
                    continue
            except Exception as e:
                print(f"读取文件时发生未知错误 {file}: {e}")
                continue
            
            try:
                # 更新像素数据
                ds.PixelData = hu_to_stored_pixels(ds, processed_ct[i]).tobytes()
                
                # 修改序列描述
                ds.SeriesDescription = series_description
                
                # 确保SOPInstanceUID唯一
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                
                # 保存修改后的DICOM，直接使用原文件名
                output_file = os.path.join(output_dicom_dir, os.path.basename(file))
                ds.save_as(output_file)
                successful_saves += 1
                
            except Exception as e:
                print(f"保存文件 {file} 时出错: {e}")
                continue
    
    print(f"{series_description}已保存到 {output_dicom_dir} (成功保存 {successful_saves}/{len(valid_dicom_files)} 个文件)")

def save_vessel_mask_as_dicom(original_dicom_dir, output_dicom_dir, vessel_mask, case_name, background_value):
    """
    将血管掩码保存为DICOM格式，保留血管结构的CT值
    """
    os.makedirs(output_dicom_dir, exist_ok=True)
    
    dicom_files = glob.glob(os.path.join(original_dicom_dir, "*.dcm"))
    if not dicom_files:
        print(f"警告: 在 {original_dicom_dir} 中未找到DICOM文件")
        return
    
    # 过滤有效的DICOM文件
    valid_dicom_files = []
    for file in dicom_files:
        try:
            # 尝试读取DICOM文件
            test_dicom = pydicom.dcmread(file, stop_before_pixels=True)
            valid_dicom_files.append(file)
        except pydicom.errors.InvalidDicomError:
            try:
                # 尝试强制读取
                test_dicom = pydicom.dcmread(file, force=True, stop_before_pixels=True)
                valid_dicom_files.append(file)
            except Exception as e:
                print(f"跳过无效文件: {file}, 错误: {e}")
                continue
    
    if not valid_dicom_files:
        print(f"警告: 在 {original_dicom_dir} 中未找到有效的DICOM文件")
        return
        
    valid_dicom_files.sort(key=dicom_sort_key)
    
    # 加载原始CT数据以获取血管区域的CT值
    original_ct, _ = load_dicom_series(original_dicom_dir)
    if original_ct is None:
        print(f"无法加载原始CT数据: {original_dicom_dir}")
        return
    
    # 创建血管CT：血管区域保留原始CT值，非血管区域设为背景值
    vessel_ct = np.full_like(original_ct, background_value, dtype=np.float32)
    vessel_ct[vessel_mask == 1] = original_ct[vessel_mask == 1]
    
    print(f"正在保存血管DICOM - 病例: {case_name}")
    successful_saves = 0
    for i, file in enumerate(tqdm(valid_dicom_files)):
        if i < vessel_ct.shape[0]:
            try:
                ds = pydicom.dcmread(file)
            except pydicom.errors.InvalidDicomError:
                try:
                    ds = pydicom.dcmread(file, force=True)
                except Exception as e:
                    print(f"无法读取文件 {file}: {e}")
                    continue
            except Exception as e:
                print(f"读取文件时发生未知错误 {file}: {e}")
                continue
            
            try:
                # 更新像素数据
                ds.PixelData = hu_to_stored_pixels(ds, vessel_ct[i]).tobytes()
                
                # 修改序列描述，标记为血管分割结果
                ds.SeriesDescription = "Vessel Segmentation"
                
                # 确保SOPInstanceUID唯一
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                
                # 保存修改后的DICOM
                output_file = os.path.join(output_dicom_dir, os.path.basename(file))
                ds.save_as(output_file)
                successful_saves += 1
                
            except Exception as e:
                print(f"保存文件 {file} 时出错: {e}")
                continue
    
    print(f"血管DICOM文件已保存到 {output_dicom_dir} (成功保存 {successful_saves}/{len(valid_dicom_files)} 个文件)")

def process_and_visualize(dicom_dir, output_dir, case_name):
    """
    处理CT数据并可视化结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载DICOM数据
    lung_ct, background_value = load_dicom_series(dicom_dir)
    if lung_ct is None:
        print(f"跳过处理: {dicom_dir}")
        return None, None, None, None
    
    print(f"处理病例: {case_name}")
    
    # 血管分割
    vessel_mask, enhanced_vessels = segment_vessels_frangi(lung_ct, background_value)
    
    # 创建处理后的CT图像（用背景值替换血管区域）
    processed_ct = lung_ct.copy()
    processed_ct[vessel_mask == 1] = background_value
    
    # 可视化关键切片
    # 选择包含更多血管的中间切片来可视化
    non_background_slices = []
    for i in range(lung_ct.shape[0]):
        if np.sum(lung_ct[i] != background_value) > 0:
            non_background_slices.append(i)
    
    if non_background_slices:
        middle_slice = non_background_slices[len(non_background_slices)//2]
        
        # 定义显示范围
        vmin = np.min(lung_ct)
        vmax = np.max(lung_ct[lung_ct != background_value])
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(lung_ct[middle_slice], cmap='gray', vmin=vmin, vmax=vmax)
        plt.title(f'原始肺部CT - 病例: {case_name}')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(enhanced_vessels[middle_slice], cmap='hot')
        plt.title('Frangi滤波增强血管')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(vessel_mask[middle_slice], cmap='gray')
        plt.title('血管掩码')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(processed_ct[middle_slice], cmap='gray', vmin=vmin, vmax=vmax)
        plt.title('去除血管后的CT')
        plt.axis('off')
        
        # 将血管掩码叠加在原始CT上显示
        plt.subplot(2, 3, 5)
        plt.imshow(lung_ct[middle_slice], cmap='gray', vmin=vmin, vmax=vmax)
        masked_vessels = np.ma.masked_where(vessel_mask[middle_slice] == 0, vessel_mask[middle_slice])
        plt.imshow(masked_vessels, cmap='hot', alpha=0.7)
        plt.title('血管分割结果叠加显示')
        plt.axis('off')
        
        # 显示血管分割的CT
        vessel_ct = np.full_like(lung_ct, background_value)
        vessel_ct[vessel_mask == 1] = lung_ct[vessel_mask == 1]
        plt.subplot(2, 3, 6)
        plt.imshow(vessel_ct[middle_slice], cmap='gray', vmin=vmin, vmax=vmax)
        plt.title('分割出的血管CT')
        plt.axis('off')
        
        # 保存可视化结果
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{case_name}_visualization.png'))
        plt.close()
        
        # 保存3D轴向视图（多切片）
        num_slices = min(6, len(non_background_slices))
        slice_indices = np.linspace(0, len(non_background_slices)-1, num_slices).astype(int)
        
        plt.figure(figsize=(15, 10))
        for i, idx in enumerate(slice_indices):
            slice_idx = non_background_slices[idx]
            plt.subplot(2, 3, i+1)
            plt.imshow(lung_ct[slice_idx], cmap='gray', vmin=vmin, vmax=vmax)
            masked_vessels = np.ma.masked_where(vessel_mask[slice_idx] == 0, vessel_mask[slice_idx])
            plt.imshow(masked_vessels, cmap='hot', alpha=0.7)
            plt.title(f'切片 #{slice_idx}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{case_name}_multi_slice_view.png'))
        plt.close()
        
        print(f"可视化结果已保存到 {output_dir}")
    else:
        print("警告：未找到包含肺部的切片")
    
    # 保存处理结果
    np.save(os.path.join(output_dir, f'{case_name}_vessel_mask.npy'), vessel_mask)
    np.save(os.path.join(output_dir, f'{case_name}_processed_ct.npy'), processed_ct)
    
    return lung_ct, vessel_mask, processed_ct, enhanced_vessels

def process_multiple_cases_with_vessel_dicom(input_root_dir, output_root_dir):
    """
    处理根目录下的多个病例，保存三种DICOM格式：
    1. 原始CT（可选）
    2. 去除血管的CT
    3. 分割出的血管CT
    """
    # 确保输出根目录存在
    os.makedirs(output_root_dir, exist_ok=True)
    
    # 获取所有子文件夹（每个子文件夹对应一个病例）
    case_folders = [f for f in os.listdir(input_root_dir) 
                    if os.path.isdir(os.path.join(input_root_dir, f))]
    
    print(f"找到 {len(case_folders)} 个病例需要处理")
    
    for case_folder in case_folders:
        case_input_dir = os.path.join(input_root_dir, case_folder)
        case_output_dir = os.path.join(output_root_dir, case_folder)
        
        # 创建子文件夹结构
        vessel_removed_dir = os.path.join(case_output_dir, "vessel_removed")
        vessel_only_dir = os.path.join(case_output_dir, "vessel_only")
        original_dir = os.path.join(case_output_dir, "original")
        
        # 检查是否已处理（检查血管文件夹）
        if os.path.exists(vessel_only_dir):
            existing_dicoms = glob.glob(os.path.join(vessel_only_dir, "*.dcm"))
            if existing_dicoms:
                print(f"跳过已处理的病例: {case_folder}")
                continue
        
        print(f"\n开始处理病例: {case_folder}")
        start_time = time.time()
        
        # 加载DICOM数据
        lung_ct, background_value = load_dicom_series(case_input_dir)
        if lung_ct is None:
            print(f"跳过处理: {case_input_dir}")
            continue
        
        print(f"处理病例: {case_folder}")
        
        # 血管分割
        vessel_mask, enhanced_vessels = segment_vessels_frangi(lung_ct, background_value)
        
        # 创建处理后的CT图像（用背景值替换血管区域）
        processed_ct = lung_ct.copy()
        processed_ct[vessel_mask == 1] = background_value
        
        # 保存三种DICOM格式
        
        # 1. 保存去除血管的CT
        save_as_dicom(case_input_dir, vessel_removed_dir, processed_ct, 
                     case_folder, "Vessel Removed CT")
        
        # 2. 保存分割出的血管CT
        save_vessel_mask_as_dicom(case_input_dir, vessel_only_dir, vessel_mask, 
                                 case_folder, background_value)
        
        # 3. 可选：复制原始DICOM文件
        # shutil.copytree(case_input_dir, original_dir)
        
        # 统计信息
        vessel_voxels = np.sum(vessel_mask)
        total_lung_voxels = np.sum(lung_ct != background_value)
        vessel_ratio = vessel_voxels / total_lung_voxels if total_lung_voxels > 0 else 0
        
        print(f"病例 {case_folder} 处理完成，耗时: {time.time() - start_time:.2f} 秒")
        print(f"检测到血管体素数: {vessel_voxels}")
        print(f"血管占肺部比例: {vessel_ratio:.4f}")
        print(f"输出文件保存在:")
        print(f"  - 去除血管CT: {vessel_removed_dir}")
        print(f"  - 血管CT: {vessel_only_dir}")

def process_multiple_cases(input_root_dir, output_root_dir):
    """
    处理根目录下的多个病例，包含可视化
    """
    # 确保输出根目录存在
    os.makedirs(output_root_dir, exist_ok=True)
    
    # 获取所有子文件夹（每个子文件夹对应一个病例）
    case_folders = [f for f in os.listdir(input_root_dir) 
                    if os.path.isdir(os.path.join(input_root_dir, f))]
    
    print(f"找到 {len(case_folders)} 个病例需要处理")
    
    for case_folder in case_folders:
        case_input_dir = os.path.join(input_root_dir, case_folder)
        case_output_dir = os.path.join(output_root_dir, case_folder)
        
        # 创建子文件夹结构
        vessel_removed_dir = os.path.join(case_output_dir, "vessel_removed")
        vessel_only_dir = os.path.join(case_output_dir, "vessel_only")
        visualization_dir = os.path.join(case_output_dir, "visualization")
        
        # 检查是否已处理
        if os.path.exists(vessel_only_dir):
            existing_dicoms = glob.glob(os.path.join(vessel_only_dir, "*.dcm"))
            if existing_dicoms:
                print(f"跳过已处理的病例: {case_folder}")
                continue
        
        print(f"\n开始处理病例: {case_folder}")
        start_time = time.time()
        
        # 处理当前病例并生成可视化
        lung_ct, vessel_mask, processed_ct, enhanced_vessels = process_and_visualize(
            case_input_dir, visualization_dir, case_folder)
        
        if lung_ct is not None:
            # 保存DICOM文件
            save_as_dicom(case_input_dir, vessel_removed_dir, processed_ct, 
                         case_folder, "Vessel Removed CT")
            
            save_vessel_mask_as_dicom(case_input_dir, vessel_only_dir, vessel_mask, 
                                     case_folder, np.min(lung_ct))
            
            # 统计信息
            vessel_voxels = np.sum(vessel_mask)
            total_lung_voxels = np.sum(lung_ct != np.min(lung_ct))
            vessel_ratio = vessel_voxels / total_lung_voxels if total_lung_voxels > 0 else 0
            
            print(f"病例 {case_folder} 处理完成，耗时: {time.time() - start_time:.2f} 秒")
            print(f"检测到血管体素数: {vessel_voxels}")
            print(f"血管占肺部比例: {vessel_ratio:.4f}")
            print(f"输出文件保存在:")
            print(f"  - 去除血管CT: {vessel_removed_dir}")
            print(f"  - 血管CT: {vessel_only_dir}")
            print(f"  - 可视化结果: {visualization_dir}")
        else:
            print(f"病例 {case_folder} 处理失败")

# 新增：只保存DICOM版本，包含血管分割
def process_multiple_cases_dicom_only(input_root_dir, output_root_dir):
    """
    处理根目录下的多个病例，只保存DICOM文件（包含血管分割）
    """
    # 确保输出根目录存在
    os.makedirs(output_root_dir, exist_ok=True)
    
    # 获取所有子文件夹（每个子文件夹对应一个病例）
    case_folders = [f for f in os.listdir(input_root_dir) 
                    if os.path.isdir(os.path.join(input_root_dir, f))]
    
    print(f"找到 {len(case_folders)} 个病例需要处理")
    
    for case_folder in case_folders:
        case_input_dir = os.path.join(input_root_dir, case_folder)
        case_output_dir = os.path.join(output_root_dir, case_folder)
        
        # 创建子文件夹结构
        vessel_removed_dir = os.path.join(case_output_dir, "vessel_removed")
        vessel_only_dir = os.path.join(case_output_dir, "vessel_only")
        
        # 检查是否已处理
        if os.path.exists(vessel_only_dir):
            existing_dicoms = glob.glob(os.path.join(vessel_only_dir, "*.dcm"))
            if existing_dicoms:
                print(f"跳过已处理的病例: {case_folder}")
                continue
        
        print(f"\n开始处理病例: {case_folder}")
        start_time = time.time()
        
        # 加载DICOM数据
        lung_ct, background_value = load_dicom_series(case_input_dir)
        if lung_ct is None:
            print(f"跳过处理: {case_input_dir}")
            continue
        
        print(f"处理病例: {case_folder}")
        
        # 血管分割
        vessel_mask, enhanced_vessels = segment_vessels_frangi(lung_ct, background_value)
        
        # 创建处理后的CT图像（用背景值替换血管区域）
        processed_ct = lung_ct.copy()
        processed_ct[vessel_mask == 1] = background_value
        
        # 保存DICOM文件
        save_as_dicom(case_input_dir, vessel_removed_dir, processed_ct, 
                     case_folder, "Vessel Removed CT")
        
        save_vessel_mask_as_dicom(case_input_dir, vessel_only_dir, vessel_mask, 
                                 case_folder, background_value)
        
        # 统计信息
        vessel_voxels = np.sum(vessel_mask)
        total_lung_voxels = np.sum(lung_ct != background_value)
        vessel_ratio = vessel_voxels / total_lung_voxels if total_lung_voxels > 0 else 0
        
        print(f"病例 {case_folder} 处理完成，耗时: {time.time() - start_time:.2f} 秒")
        print(f"检测到血管体素数: {vessel_voxels}")
        print(f"血管占肺部比例: {vessel_ratio:.4f}")
        print(f"输出文件保存在:")
        print(f"  - 去除血管CT: {vessel_removed_dir}")
        print(f"  - 血管CT: {vessel_only_dir}")

def save_vessel_removed_dicom(original_dicom_dir, output_dir, processed_ct, case_name):
    """
    将去除血管后的CT保存为DICOM格式，直接保存在输出目录中
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dicom_files = glob.glob(os.path.join(original_dicom_dir, "*.dcm"))
    if not dicom_files:
        print(f"警告: 在 {original_dicom_dir} 中未找到DICOM文件")
        return
    
    # 过滤有效的DICOM文件
    valid_dicom_files = []
    for file in dicom_files:
        try:
            # 尝试读取DICOM文件
            test_dicom = pydicom.dcmread(file, stop_before_pixels=True)
            valid_dicom_files.append(file)
        except pydicom.errors.InvalidDicomError:
            try:
                # 尝试强制读取
                test_dicom = pydicom.dcmread(file, force=True, stop_before_pixels=True)
                valid_dicom_files.append(file)
            except Exception as e:
                print(f"跳过无效文件: {file}, 错误: {e}")
                continue
    
    if not valid_dicom_files:
        print(f"警告: 在 {original_dicom_dir} 中未找到有效的DICOM文件")
        return
    
    valid_dicom_files.sort(key=dicom_sort_key)
    
    print(f"正在保存去除血管后的CT - 病例: {case_name}")
    successful_saves = 0
    for i, file in enumerate(tqdm(valid_dicom_files)):
        if i < processed_ct.shape[0]:
            try:
                ds = pydicom.dcmread(file)
            except pydicom.errors.InvalidDicomError:
                try:
                    ds = pydicom.dcmread(file, force=True)
                except Exception as e:
                    print(f"无法读取文件 {file}: {e}")
                    continue
            except Exception as e:
                print(f"读取文件时发生未知错误 {file}: {e}")
                continue
            
            try:
                # 更新像素数据
                ds.PixelData = hu_to_stored_pixels(ds, processed_ct[i]).tobytes()
                
                # 修改序列描述
                ds.SeriesDescription = "Vessel Removed CT"
                
                # 确保SOPInstanceUID唯一
                ds.SOPInstanceUID = pydicom.uid.generate_uid()
                
                # 保存修改后的DICOM，直接使用原文件名
                output_file = os.path.join(output_dir, os.path.basename(file))
                ds.save_as(output_file)
                successful_saves += 1
                
            except Exception as e:
                print(f"保存文件 {file} 时出错: {e}")
                continue
    
    print(f"去除血管的CT已保存到 {output_dir} (成功保存 {successful_saves}/{len(valid_dicom_files)} 个文件)")

def process_multiple_cases_vessel_removed_only(input_root_dir, output_root_dir):
    """
    处理根目录下的多个病例，只保存去除血管后的DICOM文件，直接保存在病例文件夹中
    """
    # 确保输出根目录存在
    os.makedirs(output_root_dir, exist_ok=True)
    
    # 获取所有子文件夹（每个子文件夹对应一个病例）
    case_folders = [f for f in os.listdir(input_root_dir) 
                    if os.path.isdir(os.path.join(input_root_dir, f))]
    
    print(f"找到 {len(case_folders)} 个病例需要处理")
    
    for case_folder in case_folders:
        case_input_dir = os.path.join(input_root_dir, case_folder)
        case_output_dir = os.path.join(output_root_dir, case_folder)
        
        # 检查是否已处理（检查输出目录中是否有DICOM文件）
        if os.path.exists(case_output_dir):
            existing_dicoms = glob.glob(os.path.join(case_output_dir, "*.dcm"))
            if existing_dicoms:
                print(f"跳过已处理的病例: {case_folder}")
                continue
        
        print(f"\n开始处理病例: {case_folder}")
        start_time = time.time()
        
        # 加载DICOM数据
        lung_ct, background_value = load_dicom_series(case_input_dir)
        if lung_ct is None:
            print(f"跳过处理: {case_input_dir}")
            continue
        
        print(f"处理病例: {case_folder}")
        
        # 血管分割
        vessel_mask, enhanced_vessels = segment_vessels_frangi(lung_ct, background_value)
        
        # 创建处理后的CT图像（用背景值替换血管区域）
        processed_ct = lung_ct.copy()
        processed_ct[vessel_mask == 1] = background_value
        
        # 保存去除血管后的DICOM文件，直接保存在病例文件夹中
        save_vessel_removed_dicom(case_input_dir, case_output_dir, processed_ct, case_folder)
        
        # 统计信息
        vessel_voxels = np.sum(vessel_mask)
        total_lung_voxels = np.sum(lung_ct != background_value)
        vessel_ratio = vessel_voxels / total_lung_voxels if total_lung_voxels > 0 else 0
        
        print(f"病例 {case_folder} 处理完成，耗时: {time.time() - start_time:.2f} 秒")
        print(f"检测到血管体素数: {vessel_voxels}")
        print(f"血管占肺部比例: {vessel_ratio:.4f}")
        print(f"去除血管的CT已保存到: {case_output_dir}")

# 修改主函数调用
if __name__ == "__main__":
    # 设置输入和输出根目录
    input_root_dir = r"F:\MosMedData\Covid\CT-0"  # 修改为你的输入根目录
    output_root_dir = r"F:\MosMedData\Covid_no_vessel\CT-0"   # 修改为你的输出根目录

    # 只保存去除血管后的DICOM文件
    process_multiple_cases_vessel_removed_only(input_root_dir, output_root_dir)
    
    print("所有病例处理完成！")
