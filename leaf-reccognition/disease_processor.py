import csv
import os
from typing import Dict, List, Optional

def load_disease_info(csv_path: str = "disease_treatment.csv") -> Dict[str, Dict[str, str]]:
    """
    从CSV文件加载病害信息，包括病害名称、问题分析和解决方案
    
    :param csv_path: CSV文件路径
    :return: 嵌套字典，格式为 {病害名称: {"问题分析": "...", "解决方案": "..."}
    """
    disease_info = {}
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"警告：CSV文件 {csv_path} 不存在，将返回空字典")
        return disease_info
    
    # 读取CSV文件
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # 定义必要的列
        required_columns = {'病害名称', '问题分析', '解决方案'}
        
        # 检查必要的列是否存在
        if not required_columns.issubset(reader.fieldnames):
            missing = required_columns - set(reader.fieldnames)
            raise ValueError(f"CSV文件缺少必要的列：{missing}")
        
        # 处理每一行数据
        for row_num, row in enumerate(reader, start=2):  # 行号从2开始（表头为第1行）
            # 去除前后空格
            disease_name = row['病害名称'].strip()
            problem_analysis = row['问题分析'].strip()
            solution = row['解决方案'].strip()
            
            # 跳过病害名称为空的行
            if not disease_name:
                print(f"警告：第{row_num}行的病害名称为空，已跳过")
                continue
            
            # 检查是否有重复的病害名称
            if disease_name in disease_info:
                print(f"警告：第{row_num}行的病害名称 '{disease_name}' 重复，将覆盖之前的记录")
            
            # 存储数据
            disease_info[disease_name] = {
                "问题分析": problem_analysis,
                "解决方案": solution
            }
    
    # print(f"成功从 {csv_path} 加载 {len(disease_info)} 条病害信息")
    return disease_info

def get_disease_details(disease_info: Dict[str, Dict[str, str]], 
                       disease_name: str) -> Optional[Dict[str, str]]:
    """
    获取指定病害的详细信息
    
    :param disease_info: 从CSV加载的病害信息字典
    :param disease_name: 要查询的病害名称
    :return: 包含问题分析和解决方案的字典，若未找到则返回None
    """
    return disease_info.get(disease_name)

# 使用示例
if __name__ == "__main__":
    # 加载病害信息
    disease_data = load_disease_info()
    
    # 测试查询
    # test_diseases = [
    #     "apple healthy（苹果健康）",
    #     "Apple_Scab general（苹果黑星病一般）",
    #     "未知病害"
    # ]
    
    # for disease in test_diseases:
    #     print(f"=== {disease} ===")
    #     details = get_disease_details(disease_data, disease)
        
    #     if details:
    #         print(f"问题分析：{details['问题分析']}")
    #         print(f"解决方案：{details['解决方案']}")
    #     else:
    #         print("未找到该病害的信息")
    #     print()  # 空行分隔
