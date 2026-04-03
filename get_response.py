from langchain_core.messages import HumanMessage
import random
import config
import json
from typing import List, Dict, Any, Tuple, Optional
import get_database
from small_models_library import FUNCTION_MAP

def get_ps_by_process_id(process_id: str, ps_json_path: str = config.PS_Url) -> Dict[str, Any]:
    """
    Retrieve the process scheduling (PS) configuration by process ID.

    Args:
        process_id: The process agent ID (e.g., "P001")
        ps_json_path: Path to the PS configuration JSON file

    Returns:
        Dictionary containing the PS configuration

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the JSON file is malformed
        KeyError: If the process_id is not found in the configuration
    """
    try:
        with open(ps_json_path, 'r', encoding='utf-8') as f:
            ps_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {ps_json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}")

    if process_id not in ps_data:
        raise KeyError(f"Process scheduling model not found for process_id '{process_id}'")

    return ps_data[process_id]

def validator(ps_config: Dict[str, Any], queried_data: Dict[str, Any]) -> Tuple[bool, Optional[List[str]], Optional[Dict[str, Any]]]:
    """
    Validate that all required PAM input points are present in the queried data.

    Args:
        ps_config: PS configuration dictionary containing 'pam_input_points'
        queried_data: Dictionary of data retrieved from database (point_id -> value)

    Returns:
        Tuple: (is_valid, missing_points_list, complete_data_dict)
    """
    required = ps_config.get("pam_input_points", [])
    missing = [pt for pt in required if pt not in queried_data]
    if missing:
        return False, missing, None
    complete_data = {pt: queried_data[pt] for pt in required}
    return True, None, complete_data

def executor(ps_config: Dict[str, Any], external_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the small model nodes in sequence according to the PS configuration.
    Handles parent-child data dependencies.

    Args:
        ps_config: PS configuration containing the list of nodes
        external_data: Validated external input data (point_id -> value)

    Returns:
        Dictionary of all outputs from the executed nodes

    Raises:
        RuntimeError: If a parent node hasn't executed, function is unknown, or node execution fails
    """
    nodes = ps_config.get("nodes", [])
    node_outputs = {}
    for node in nodes:
        node_id = node["node_id"]
        parent_input = {}
        for parent_id in node.get("parents", []):
            if parent_id not in node_outputs:
                raise RuntimeError(f"Parent node {parent_id} has not been executed")
            parent_input.update(node_outputs[parent_id])
        required_external = node.get("external_inputs", [])
        node_external = {k: external_data.get(k) for k in required_external if k in external_data}
        func_name = node.get("function")
        if func_name not in FUNCTION_MAP:
            raise RuntimeError(f"Unknown function name: {func_name}")
        func = FUNCTION_MAP[func_name]
        try:
            output = func(node_external, parent_input)
            node_outputs[node_id] = output
        except Exception as e:
            raise RuntimeError(f"Execution failed for node {node_id}: {e}")
    final_result = {}
    for out in node_outputs.values():
        final_result.update(out)
    print(final_result)
    return final_result

from langchain_core.messages import HumanMessage

def generate_feedback(sm_result: dict, task_description: str, chat_model, parsed_input: dict,
                      output_description: dict = None):
    """
    Generate natural language feedback from small model results.

    Args:
        sm_result: Dictionary of small model outputs (point_id -> value)
        task_description: User's original query
        parsed_input: Additional user requirements
        output_description: Optional mapping from point_id to human-readable description
    """
    # Convert results to readable format
    if output_description is None:
        output_description = {}
    readable_result = {}
    for k, v in sm_result.items():
        desc = output_description.get(k, k)
        readable_result[desc] = v

    feedback_prompt = f"""
    根据以下任务描述和小模型结果，生成自然语言反馈：
    - 用户的提问：{task_description}
    - 用户的其他要求：{parsed_input}
    - 小模型计算的结果（已附上含义说明）：
    {readable_result}

    请用流畅、专业的中文回复，必须基于上述事实，不要编造。
    """
    feedback = chat_model.invoke([HumanMessage(content=feedback_prompt)]).content
    return feedback

def get_response(process_id, chat_model="", user_message="", parsed_input=""):
    """
    Main controller entry point. Coordinates validator, executor, and generator.

    Args:
        process_id: Process agent ID (e.g., "P001")
        chat_model: Language model instance for feedback generation
        user_message: User's natural language input
        parsed_input: Additional user requirements

    Returns:
        Natural language response string
    """
    response = ""
    # Step 1: Retrieve PS configuration by process_id
    try:
        ps_config = get_ps_by_process_id(process_id)
    except (FileNotFoundError, KeyError, ValueError) as e:
        response = f"Error: Unable to load scheduling model - {str(e)}"
        return response

    required_points = ps_config.get("pam_input_points", [])
    if not required_points:
        response = "Error: PS configuration missing 'pam_input_points'"
        return response

    # Query database for the required points
    queried_data = get_database.query_database(required_points, missing_probability=0.0)

    passed, missing, complete_data = validator(ps_config, queried_data)
    if not passed:
        missing_info = "\n".join(f"  - {pt}" for pt in missing)
        response = f"Incomplete data: missing the following measurement points:\n{missing_info}"
        return response

    # Execute the scheduling model
    execution_result = executor(ps_config, complete_data)

    # Prepare output description using configured point descriptions
    output_points = ps_config.get("pam_output_points", [])
    output_desc = {pt: config.POINT_DESCRIPTION.get(pt, pt) for pt in output_points}

    # Generate natural language feedback
    response = generate_feedback(
        sm_result=execution_result,
        task_description=user_message,
        chat_model=chat_model,
        parsed_input=parsed_input or {},
        output_description=output_desc
    )
    return response

# get_response("P001")