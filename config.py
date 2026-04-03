
# parameter configuration

# Path to the PAMs (Process Agent Models) configuration file
PAMs_Url = "PAMs.json"

# Path to the PS (Process Scheduling) configuration file
PS_Url = "PS.json"

# Path to the dataset Excel file for rolling mill gearbox QA
Dataset_Url = "Rolling_mill_gearbox_QA_dataset.xlsx"

# API key for accessing the language model service (e.g., Deepseek)
api_key = "------"

# Path to the paraphrase model (MiniLM) for text vectorization
paraphrase_Url = "paraphrase-multilingual-MiniLM-L12-v2"

# Path to the pre-trained XGBoost classifier model for matching score regression
XGBoost_Url = "best_classifier_model.pkl"

# Mapping from measurement point IDs to human-readable descriptions for output generation
POINT_DESCRIPTION = {
    "40PG20251030001": "健康状态评估结果",
    "40YC20251030001": "健康状态评估分数",
    "40ZD20251030002_out": "齿轮箱故障诊断结果",
    "40JG20251030002": "齿轮箱故障类型",
    "40OL20251030003": "齿轮箱运行状态",
    "40FL20251030003": "齿轮箱状态预测",
    "40EL20251030004": "剩余寿命预测结果",
    "40OL20251030004": "齿轮箱故障预测结果",
    "40OL20251030005": "运维决策方案",
    "40OL20251030006": "运维优化建议",
}
