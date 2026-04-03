import random
from typing import List, Dict, Any

def query_database(point_ids: List[str], missing_probability: float = 0.0) -> Dict[str, Any]:
    """
    Simulate querying measurement point data from a database.
    Dynamically generates random mock values based on the measurement point ID.

    Args:
        point_ids: List of measurement point IDs to query
        missing_probability: Probability (0~1) of a point being missing, used for testing validator

    Returns:
        Dictionary mapping point IDs to values (may be incomplete if missing_probability > 0)
    """
    result = {}
    for pid in point_ids:
        # Simulate data missing
        if random.random() < missing_probability:
            continue

        # Generate a random value based on the point ID pattern
        value = _generate_mock_value(pid)
        result[pid] = value
    return result

def _generate_mock_value(pid: str) -> Any:
    """Generate a single mock value based on the measurement point ID."""
    # Vibration data (contains ZD)
    if "ZD" in pid:
        vib = round(random.uniform(5.0, 20.0), 1)
        return f"{vib} mm/s"

    # Temperature data (contains WD)
    if "WD" in pid:
        temp = random.randint(70, 105)
        return f"{temp} °C"

    # Lubrication data (contains RR)
    if "RR" in pid:
        return random.choice(["良好", "一般", "差"])

    # Wear data (contains MS)
    if "MS" in pid:
        return random.choice(["无磨损", "轻微磨损", "中度磨损", "严重磨损"])

    # Rotational speed data (contains ZZ)
    if "ZZ" in pid:
        rpm = random.randint(1000, 2000)
        return f"{rpm} rpm"

    # Load data (contains FZ)
    if "FZ" in pid:
        load = random.randint(30, 100)
        return f"{load}%"

    # Gear size (contains CC)
    if "CC" in pid:
        return random.choice(["标准", "小型", "大型", "定制"])

    # Environmental data (contains HJ)
    if "HJ" in pid:
        return random.choice(["正常", "高温", "高湿", "粉尘"])

    # Continuous data: vibration sequence (LZ), temperature sequence (LW), speed sequence (ZS), wear sequence (LM)
    if pid.startswith("40LZ"):  # Continuous vibration data
        length = random.randint(3, 6)
        values = [round(random.uniform(8.0, 18.0), 1) for _ in range(length)]
        return values
    if pid.startswith("40LW"):  # Continuous temperature data
        length = random.randint(3, 6)
        values = [random.randint(75, 98) for _ in range(length)]
        return values
    if pid.startswith("40ZS"):  # Continuous speed data
        length = random.randint(3, 6)
        base = random.randint(1400, 1600)
        values = [base + random.randint(-30, 30) for _ in range(length)]
        return values
    if pid.startswith("40LM"):  # Continuous wear data
        length = random.randint(3, 6)
        values = [round(random.uniform(0.05, 0.25), 3) for _ in range(length)]
        return values

    # Default: generate a random string value
    return f"模拟数据_{pid}_{random.randint(1, 100)}"