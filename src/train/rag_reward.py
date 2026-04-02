import re

def compute_score(data_source, solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    solution_str: 大模型生成的整段包含思维链和搜索的文本
    ground_truth: 训练集 Parquet 中包含的正确答案数据
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    
    if not answer_match:
        # 模型未能遵循指令输出 <answer>，给予轻微惩罚
        return -0.5 
        
    extracted_answer = answer_match.group(1).strip()
    target = ground_truth.get("target", "")

    # 判断 target 字符串或列表是否包含在生成的答案中
    if isinstance(target, list):
        if any(t in extracted_answer for t in target):
            return 1.0
    else:
        if target in extracted_answer:
            return 1.0
            
    # 回答错误
    return 0.0