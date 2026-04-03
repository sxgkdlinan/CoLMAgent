import json
import random
from typing import Dict, Any, List, Tuple, Optional

def feature_extraction_for_health(external: Dict, parent: Dict) -> Dict:
    """健康特征提取：从外部测点计算健康特征"""
    vib_str = external.get("40ZD20251030001", "0")
    temp_str = external.get("40WD20251030001", "0")
    lube = external.get("40RR20251030001", "良好")
    wear = external.get("40MS20251030001", "轻微磨损")
    vib = float(vib_str.replace("mm/s", ""))
    temp = float(temp_str.replace("°C", ""))
    vib_feature = min(max(vib / 20.0, 0), 1)
    temp_feature = min(max((temp - 70) / 30.0, 0), 1)
    lube_score = 1.0 if lube == "良好" else 0.5 if lube == "一般" else 0.2
    wear_score = 1.0 if wear == "无磨损" else 0.7 if wear == "轻微磨损" else 0.3
    health_feature = (vib_feature + temp_feature + lube_score + wear_score) / 4
    return {"health_feature_internal": health_feature}

def health_score_calculator(external: Dict, parent: Dict) -> Dict:
    """健康评分：基于内部特征计算最终分数和状态"""
    feature = parent.get("health_feature_internal", 0.5)
    score = int(feature * 100)
    if score >= 80:
        status = "运行状态正常"
    elif score >= 60:
        status = "运行状态注意"
    else:
        status = "运行状态警告"
    return {"40PG20251030001": status, "40YC20251030001": str(score)}

# P002 节点
def fault_feature_extraction(external: Dict, parent: Dict) -> Dict:
    """故障特征提取"""
    vib_str = external.get("40ZD20251030002", "0")
    temp_str = external.get("40WD20251030002", "0")
    vib = float(vib_str.replace("mm/s", ""))
    temp = float(temp_str.replace("°C", ""))
    vib_norm = min(max(vib / 15.0, 0), 1)
    temp_norm = min(max((temp - 75) / 25.0, 0), 1)
    return {"vib_norm": vib_norm, "temp_norm": temp_norm}

def fault_classifier(external: Dict, parent: Dict) -> Dict:
    """故障分类"""
    vib_norm = parent.get("vib_norm", 0)
    temp_norm = parent.get("temp_norm", 0)
    if vib_norm > 0.8 or temp_norm > 0.7:
        diagnosis = random.choice(["中度故障", "严重故障"])
        fault_type = random.choice(["齿轮磨损", "轴承损坏", "润滑不良"])
    elif vib_norm > 0.4:
        diagnosis = "轻微故障"
        fault_type = random.choice(["齿轮磨损", "轴承损坏"])
    else:
        diagnosis = "无故障"
        fault_type = "无故障"
    return {"40ZD20251030002_out": diagnosis, "40JG20251030002": fault_type}

# P003 节点
def state_estimator(external: Dict, parent: Dict) -> Dict:
    """状态估计器"""
    vib_str = external.get("40ZD20251030003", "0")
    temp_str = external.get("40WD20251030003", "0")
    vib = float(vib_str.replace("mm/s", ""))
    temp = float(temp_str.replace("°C", ""))
    if vib > 10 or temp > 90:
        status = "中度异常"
    elif vib > 5 or temp > 80:
        status = "轻微异常"
    else:
        status = "正常运行"
    return {"current_status_internal": status}

def state_predictor(external: Dict, parent: Dict) -> Dict:
    """状态预测"""
    current = parent.get("current_status_internal", "正常运行")
    if current == "正常运行":
        pred = random.choices(["稳定运行", "需关注"], weights=[0.8, 0.2])[0]
    elif current == "轻微异常":
        pred = random.choices(["需关注", "建议检修"], weights=[0.6, 0.4])[0]
    else:
        pred = random.choices(["建议检修", "紧急维修"], weights=[0.5, 0.5])[0]
    return {"40OL20251030003": current, "40FL20251030003": pred}

# P004 节点
def degradation_analysis(external: Dict, parent: Dict) -> Dict:
    """退化趋势分析"""
    vib_seq = external.get("40LZ20251030004", [10,12,15])
    temp_seq = external.get("40LW20251030004", [80,82,85])
    if isinstance(vib_seq, str):
        vib_seq = [float(x) for x in vib_seq.strip("[]").split(",")]
    if isinstance(temp_seq, str):
        temp_seq = [float(x) for x in temp_seq.strip("[]").split(",")]
    if len(vib_seq) > 1:
        vib_trend = (vib_seq[-1] - vib_seq[0]) / len(vib_seq)
    else:
        vib_trend = 0
    if len(temp_seq) > 1:
        temp_trend = (temp_seq[-1] - temp_seq[0]) / len(temp_seq)
    else:
        temp_trend = 0
    deg_index = (vib_trend/5 + temp_trend/10) / 2
    deg_index = max(0, min(deg_index, 1))
    return {"degradation_index": deg_index}

def rlf_predictor(external: Dict, parent: Dict) -> Dict:
    """剩余寿命预测"""
    deg = parent.get("degradation_index", 0)
    if deg < 0.05:
        life = ">10000小时"
        risk = "无故障风险"
    elif deg < 0.15:
        life = "5000-10000小时"
        risk = "低风险"
    elif deg < 0.3:
        life = "2000-5000小时"
        risk = "中等风险"
    else:
        life = "<1000小时"
        risk = "高风险"
    return {"40EL20251030004": life, "40OL20251030004": risk}

# P005 节点
def risk_assessment(external: Dict, parent: Dict) -> Dict:
    """风险评估"""
    vib_seq = external.get("40LZ20251030006", [10,12,15])
    temp_seq = external.get("40LW20251030006", [80,82,85])
    if isinstance(vib_seq, str):
        vib_seq = [float(x) for x in vib_seq.strip("[]").split(",")]
    if isinstance(temp_seq, str):
        temp_seq = [float(x) for x in temp_seq.strip("[]").split(",")]
    avg_vib = sum(vib_seq)/len(vib_seq) if vib_seq else 0
    avg_temp = sum(temp_seq)/len(temp_seq) if temp_seq else 0
    if avg_vib > 12 or avg_temp > 88:
        risk = 3
    elif avg_vib > 8 or avg_temp > 82:
        risk = 2
    else:
        risk = 1
    return {"risk_level_internal": risk}

def decision_generator(external: Dict, parent: Dict) -> Dict:
    """决策方案生成"""
    risk = parent.get("risk_level_internal", 1)
    if risk >= 3:
        plan = "立即停机检修"
        advice = "全面检查齿轮箱，更换损坏部件"
    elif risk == 2:
        plan = "计划性维护，1周内安排"
        advice = "加强监测，准备备件"
    else:
        plan = "继续运行，定期监测"
        advice = "优化润滑周期，每500小时检查一次"
    return {"40OL20251030005": plan, "40OL20251030006": advice}

# 函数映射表（与PS.json中的function字段对应）
FUNCTION_MAP = {
    "feature_extraction_for_health": feature_extraction_for_health,
    "health_score_calculator": health_score_calculator,
    "fault_feature_extraction": fault_feature_extraction,
    "fault_classifier": fault_classifier,
    "state_estimator": state_estimator,
    "state_predictor": state_predictor,
    "degradation_analysis": degradation_analysis,
    "rlf_predictor": rlf_predictor,
    "risk_assessment": risk_assessment,
    "decision_generator": decision_generator,
}