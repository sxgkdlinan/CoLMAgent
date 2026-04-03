from langchain_core.messages import HumanMessage
import json
import logging
from typing import Dict, Optional, List
import re


# Parse the user's input to extract the task type, device object, input data, expected output, and other restrictive requirements.

def parse_user_input(chat_model, user_input: str) -> Dict[str, Optional[str]]:

    # Prompt words for large language models
    parse_prompt = f"""
    请从以下用户输入中提取以下字段：
    1. 任务类型，任务类型包括但不限于：健康评估、 健康状态、状态评分、故障诊断、故障判断、故障检测、数字孪生、状态预测、运行状态、状态监测、预测性维护、故障预测、剩余寿命预测、运维决策、优化策略、维护方案、维护策略
    2. 任务对象：对象名称、对象编号，例如：齿轮箱CLX001, 齿轮箱内圈齿轮Gear001, 齿轮箱外圈齿轮Gear002, 齿轮箱轴承Bearing001。如果没有提取到任务对象，则默认是齿轮箱
    3. 输入数据：输入数据是什么
    4. 期望输出：期望得到的结果，包括但不限于：健康状态评估结果、健康状态评估得分、齿轮箱故障诊断结果、齿轮箱故障类型、齿轮箱运行状态、齿轮箱状态预测、剩余寿命预测结果、齿轮箱故障预测结果、运维决策方案、运维优化建议
    5. 其他限制要求

    用户输入：
    {user_input}

    请严格按照以下 JSON 格式返回解析结果，不要包含任何特殊字符（如单双引号）和 Markdown 标记：
    {{
        "task": "任务类型",
        "device_name": "对象名称",
        "device_id": "对象id",
        "input_data": "输入数据",
        "output_data": "期望输出",
        "constraints": "其他限制要求"
    }}

    示例：
    用户输入：我想诊断一下齿轮箱CLX001的现在存在故障情况，要求准确。
    解析结果：
    {{
        "task": "故障诊断",
        "device_name": "齿轮箱",
        "device_id": "CLX001"
        "input_data": "实时数据",
        "output_data": "齿轮箱故障诊断结果",
        "constraints": "准确"
    }}
    """

    try:
        # Invoke the large model to parse the user input.
        parsed_result = chat_model.invoke([HumanMessage(content=parse_prompt)]).content
        logging.info("The original output of the large model (unformatted): \n%s", repr(parsed_result))

        # 清理字符串
        parsed_result = parsed_result.strip()
        parsed_result = re.sub(r'\s+', ' ', parsed_result)
        parsed_result = parsed_result.replace("'", '"')

        # 验证 JSON 格式
        json.loads(parsed_result)  # 验证是否有效
        logging.info("The JSON format returned by the large model is valid.")

        # 将字符串解析为 JSON 对象（字典）
        parsed_dict = json.loads(parsed_result)
        return parsed_dict
    except json.JSONDecodeError as e:
        logging.error("The JSON format returned by the large model is invalid. Error message: %s", str(e))
        logging.error("Invalid JSON content: %s", parsed_result)
        return {
            "task": None,
            "device_name": None,
            "device_id": None,
            "input_data": None,
            "output_data": None,
            "constraints": None
        }
    except Exception as e:
        logging.error("An error occurred while parsing the user's input: %s", str(e))
        return {
            "task": None,
            "device_name": None,
            "device_id": None,
            "input_data": None,
            "output_data": None,
            "constraints": None
        }
