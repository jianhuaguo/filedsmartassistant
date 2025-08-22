import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def xml_to_yolo(xml_dir, img_dir, output_dir):
    """
    将VOC格式的XML标注转换为YOLOv8所需的TXT格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有XML文件
    for xml_file in tqdm(os.listdir(xml_dir)):
        if not xml_file.endswith('.xml'):
            continue
            
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"解析{xml_file}出错: {e}")
            continue
        
        # 获取图片尺寸
        size = root.find('size')
        if size is None:
            print(f"{xml_file}缺少size信息")
            continue
            
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 准备输出的TXT文件
        txt_file = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_file)
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            # 处理每个目标
            for obj in root.iter('object'):
                # 获取类别ID
                cls = obj.find('name').text
                if not cls.isdigit():
                    print(f"{xml_file}中的类别不是数字: {cls}")
                    continue
                cls_id = int(cls)
                
                # 获取边界框
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                    
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 转换为YOLO格式 (归一化中心坐标和宽高)
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                # 确保值在0-1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w = max(0, min(1, w))
                h = max(0, min(1, h))
                
                # 写入文件 (格式: class_id x_center y_center w h)
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

if __name__ == "__main__":
    # 配置路径
    XML_DIR = "data/annotations"       # XML标注目录
    IMG_DIR = "data/images"           # 图片目录
    OUTPUT_DIR = "datasets/temp/labels"  # 转换后的标签目录
    
    # 执行转换
    xml_to_yolo(XML_DIR, IMG_DIR, OUTPUT_DIR)
    print("XML转YOLO格式完成!")
