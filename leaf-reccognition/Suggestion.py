import os 
from inference import predict 
from disease_processor import load_disease_info, get_disease_details
# 加载病害数据
disease_data = load_disease_info()


def simulate_model_output(image_path=None):
    """
    模拟模型输出：调用predict()函数获取病害名称
    :param image_path: 图片路径（可选，默认使用测试图片）
    :return: 模型预测的病害名称（如 "Apple_Scab general"）
    """
    # 如果没传图片路径，使用默认测试图片（请替换为你的实际测试图片路径）
    if image_path is None:
        image_path = rf"data\train\52\fe3c6ff0cfcb699dcadf6e6cc1d2d057.jpg"
     # 检查图片是否存在
    if not os.path.isfile(image_path):
        return f"错误：图片路径不存在 - {image_path}"
    # 调用已有的predict()函数获取病害名称
    disease_name = predict(image_path)
    return disease_name


def get_treatment_details():
    """获取完整的病害分析和处理方案"""
    # 1. 获取模型预测的病害名称
    disease_name = simulate_model_output(rf"data\train\58\0bf3eb4c-e2cd-4e72-8834-4fc045bd67ae___PSU_CG 2414.JPG")
    
    # 2. 如果模型返回错误（如路径不存在），直接返回错误信息
    if disease_name.startswith("错误："):
        return {
            "病害名称": None,
            "问题分析": None,
            "解决方案": disease_name
        }
    
    # 3. 从CSV数据中查询详细信息
    details = get_disease_details(disease_data, disease_name)
    
    if details:
        return {
            "病害名称": disease_name,
            "问题分析": details["问题分析"],
            "解决方案": details["解决方案"]
        }
    else:
        return {
            "病害名称": disease_name,
            "问题分析": None,
            "解决方案": "暂未收录该病害的处理方案，请联系技术支持补充。"
        }

if __name__ == "__main__":
    # 执行查询并获取结果
    result = get_treatment_details()
    
    # 打印结果（按需要格式化输出）
    print(f"识别病害：{result['病害名称'] or '未知'}")
    if result["问题分析"]:
        print(f"问题分析：{result['问题分析']}")
    print(f"处理方案：{result['解决方案']}")



