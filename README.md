### 关于打包格式
<img width="914" height="218" alt="image" src="https://github.com/user-attachments/assets/f615664b-e926-4218-ba3c-e5ca99b0082e" />
Byte1 的 [7:2] 会立刻被 Pixel1 的高 6 位填满，而 Pixel1 剩余的低 2 位则继续挤进 Byte2 的低位……以此类推。它们像履带一样紧密咬合，直到凑满 5 个字节。

### 关于解马赛克
可以直接调用OpenCV的函数
```python
raw_data = open(input_filename, 'rb').read()
# 转换为numpy数组并转换端序
data = np.frombuffer(raw_data, dtype=np.uint16) # 小端序dtype='<u2'
data = data.byteswap()
# 重塑为图像尺寸
data = data.reshape(height, width)
# 直接使用numpy数组进行去马赛克
demosaiced = bilinear_demosaic(data)
# 应用伪彩色映射（COLORMAP_HOT, JET, etc.）强化显示效果
# colored = cv2.applyColorMap(cv2.convertScaleAbs(demosaiced, alpha=1), cv2.COLORMAP_SPRING)
# 转换为16位整数
demosaiced_uint16 = np.clip(demosaiced, 0, 65535).astype(np.uint16)
# 转换为PIL Image并保存
img = Image.fromarray(demosaiced_uint16, mode='I;16')
img.save(output_pngname)
```
