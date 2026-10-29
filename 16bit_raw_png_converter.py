"""
注意：
完整的raw处理至少要考虑:
1.每像素位深（8/10/12/14/16）与容器格式（unpacked 16-bit / packed RAW10/12 / 8-bit）
2.字节序（little / big endian）
3.对齐方式（低位对齐 LSB-aligned / 高位对齐 MSB-aligned）
4.通道布局（单通道灰度 / Bayer pattern / planar RGB / packed RGB）
5.header 长度 / 数据偏移
6.行 stride（是否含 padding）
7.缩放策略（归一化到 16-bit 的移位量应由位深推导：shift = 16 - bitdepth，MSB 对齐时 shift=0）

此程序处理的raw，每像素占用两字节，位深为10bit，字节序为小端，对齐方式为LSB，bayer阵列为RGGB，无header
读取png时，头文件( IHDR 块)已经包含宽、高、通道数、像素位深、字节序等信息，不需要像 raw 格式那样另外指定
raw_to_png 保存的是单通道 16-bit 的PNG——raw_to_png 与 png_to_raw 两个函数互不正逆
png_to_raw 只能应对灰度图与RGB/RGBA的PNG
"""
from PIL import Image, ImageDraw
import numpy as np
import os
import cv2
from scipy.signal import convolve2d
def raw_to_png(input_filename, width, height, output_pngname):
    """
    处理流程：
        1. 以指定宽高和像素位深读取raw图
        2. 转换为numpy数组
        3. reshape
        4. 转换为小端序字节流
        5. 写入二进制文件
    """

    raw_data = open(input_filename, 'rb').read()
    data = np.frombuffer(raw_data, dtype=np.uint16) # 系統預設（通常是小端）
    # data1 = np.frombuffer(raw_data, dtype='>u2')   # 明确指定大端'<u2'
    # data2 = np.frombuffer(raw_data, dtype='<u2')   # 明确指定小端'<u2'
    print(data.dtype.byteorder)  # 查看端序
    # data = data.byteswap() # 如果raw原本是大端，由于读取默认是小端，需交换回大端序
    # 检查是否某些bit从未使用(确定位深)
    # for bit in range(16):
    #     bit_set = (data >> bit) & 1
    #     print(f"Bit {bit:2d}: {bit_set.sum()} / {len(data)} pixels set")
    data = data.reshape(height, width)
    # 缩放策略（归一化到 16-bit 的移位量由前面打印的位深信息推导：shift = 16 - bitdepth；MSB 对齐时 shift=0）
    # 此处假设数据是10位，需左移6位，缩放为int16，相当于65535*value/(2^10)    
    data_16bit = data << 6 
    cv2.imwrite(output_pngname, data_16bit )

def png_to_raw(png_path, raw_output_path):
    """
    处理流程：
        1. 读取16位大端序PNG
        2. 转换为numpy数组
        3. 移除单通道维度
        4. 保存为为16bit小端序字节流
        5. 写入二进制文件
    """
    # 读取16位大端序图像
    img = Image.open(png_path)
    img_array = np.array(img)  # shape=(height, width)
    if img_array.ndim == 2: # 灰度图
        arr = img_array
    elif img_array.ndim == 3: # RGBA or RGB 取第一个通道
        gray = img_array[..., 0]
        if gray.dtype == np.uint8:
            # 8-bit → 16-bit：先放大再取整（255*257=65535）
            arr = gray.astype(np.uint16) * 257
        elif gray.dtype == np.uint16:
            arr = gray
        elif gray.dtype == np.int32:
            arr = np.clip(gray, 0, 65535).astype(np.uint16)
        else:
            # float32/64 等：先取整再截断，避免向下截断和负数回绕
            arr = np.clip(np.round(gray.astype(np.float64)), 0, 65535).astype(np.uint16)
    else:
        print("unsupported type of png")
        return
    arr.astype('<u2').tofile(raw_output_path) # 保存为小端序
    print(f"成功转换 {os.path.basename(png_path)} 为RAW格式")

if __name__== "__main__":
    # raw_folder = input('please input png/raw folder path:\n')
    png_folder = input('please input png/raw folder path:\n')
    width = int(input('please input raw image width:\n'))
    height = int(input('please input raw image height:\n'))    
    raw_folder = r'D:\...\ConverterFolder'
    width = 4224
    height = 3024
    convert_mode = 'png2raw'
    if convert_mode == 'png2raw':
        for filename in os.listdir(png_folder):
            # 检查文件扩展名是否为.png
            if filename.endswith('.png'):
                parts = filename.split('.')
                input_filename = os.path.join(png_folder, filename)
                output_rawname = os.path.join(png_folder, f"{parts[0]}.raw")
                png_to_raw(input_filename,output_rawname)
                print('processing ' + filename)
    else:
        for filename in os.listdir(raw_folder):
            # 检查文件扩展名是否为.raw
            if filename.endswith('.raw'):
                parts = filename.split('.')
                input_filename = os.path.join(raw_folder, filename)
                output_pngname = os.path.join(raw_folder, f"{parts[0]}.png")
                raw_to_png(input_filename, width, height, output_pngname)
                print('processing ' + filename)

# 附加解马赛克函数，需要根据Bayer阵列信息来制定，这里假定为RGGB，使用邻域平均             
def focused_pixel_compensation(data):
    """
    对指定位置data(5:8:h,1:4:w)与data(1:8:h,3:4:w)的像素取一个 5×5 邻域，❌
    对指定位置data(2:4:h,0:8:w)与data(0:4:h,4:8:w)的像素取一个 5×5 邻域，✅
    此处需要注意matlab索引从1开始，python从0开始；并且二维数组的索引是y坐标在前，x坐标在后
    然后只取邻域中 (0,2)、(4,2)、(2,0)、(2,4) 四个位置的像素值，最后求均值，用均值替代原像素值
    """
    h, w = data.shape
    result = data.copy().astype(np.float64)

    # 使用meshgrid确保行列数量匹配
    # start:stop:step 生成第一组位置
    rows1 = np.arange(2, h, 4)
    cols1 = np.arange(0, w, 8)
    r1_grid, c1_grid = np.meshgrid(rows1, cols1, indexing='ij')
    r1 = r1_grid.ravel()
    c1 = c1_grid.ravel()

    # 生成第二组位置
    rows2 = np.arange(0, h, 4)
    cols2 = np.arange(4, w, 8)
    r2_grid, c2_grid = np.meshgrid(rows2, cols2, indexing='ij')
    r2 = r2_grid.ravel()
    c2 = c2_grid.ravel()

    # 合并所有位置
    rows = np.concatenate([r1, r2]) #全部y坐标
    cols = np.concatenate([c1, c2]) #全部x坐标

    # # 验证长度匹配
    # print(f"位置数量: {len(rows)}")
    # print(f"rows长度: {len(rows)}, cols长度: {len(cols)}")
    # assert len(rows) == len(cols), "rows和cols长度不匹配"

    # 过滤边界
    valid = (rows >= 2) & (rows < h - 2) & (cols >= 2) & (cols < w - 2)
    rows, cols = rows[valid], cols[valid]

    # 采样点偏移（十字形邻域）
    dr = np.array([0, 4, 2, 2]) - 2 #[-2, 2, 0, 0]
    dc = np.array([2, 2, 0, 4]) - 2 #[0, 0, -2, 2]

    # 批量提取值，二维数组索引时，第一个参数就是行
    # rows是所有有效的y坐标，形状(N, )的一维向量。rows[:, None]把它扩展成(N, 1)矩阵（以便进行广播操作）。
    # 加上dr(形状(4, ))会广播成(N, 4)，表示每个点的4个偏移行坐标。
    # values.mean(axis=1) → 对每个点的 4 个邻域值取平均，得到 (N,)。然后把这个平均值写入 result 的对应位置 (rows, cols)
    if len(rows) > 0:
        # temp = rows[:, None] + dr
        values = data[rows[:, None] + dr, cols[:, None] + dc]
        result[rows, cols] = values.mean(axis=1)

    return result
