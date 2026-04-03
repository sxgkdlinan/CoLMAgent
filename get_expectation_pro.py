import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import jieba
from typing import List, Dict, Any
import config

# 全局匹配器实例
_semantic_matcher = None

def get_semantic_matcher():
    """Get the singleton semantic matcher"""
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = GearboxSemanticMatcher()
    return _semantic_matcher

class GearboxSemanticMatcher:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the gearbox operation and maintenance semantic matcher
        """
        print(f"Loading semantic matching model: {model_name}")
        self.model = SentenceTransformer(config.paraphrase_Url)

        # Specialized dictionary for gearbox operation and maintenance domain
        self.domain_dict = {
            "健康评估": ["健康评价", "健康诊断", "健康分析", "健康状态评估", "健康度评估"],
            "健康状态": ["健康状况", "健康水平", "健康度", "设备健康", "状态健康"],
            "状态评分": ["状态打分", "状态评级", "健康评分", "评估分数", "状态等级"],
            "故障诊断": ["故障分析", "故障检测", "故障判断", "故障定位", "故障识别"],
            "故障判断": ["故障判定", "故障确定", "故障识别", "故障诊断", "故障分析"],
            "故障检测": ["故障发现", "故障识别", "故障监测", "故障探测", "故障察觉"],
            "数字孪生": ["数字双胞胎", "数字化映射", "虚拟孪生", "数字镜像", "数字模型"],
            "实时状态": ["当前状态", "即时状态", "实时运行状态", "当前运行状态"],
            "运行状态": ["工作状态", "运转状态", "操作状态", "运行工况", "运行模式"],
            "状态监测": ["状态监控", "状态检测", "状态监视", "状态观察", "状态追踪"],
            "预测性维护": ["预见性维护", "预防性维护", "预测维护", "基于状态的维护"],
            "故障预测": ["故障预警", "故障预报", "故障预判", "故障预测分析"],
            "剩余寿命预测": ["剩余使用寿命预测", "剩余寿命估计", "寿命预测", "RUL预测"],
            "运维决策": ["运维决策制定", "运维决策支持", "运维方案决策", "维护决策"],
            "优化策略": ["优化方案", "优化措施", "优化方法", "最佳策略", "最优策略"],
            "维护方案": ["维护计划", "维护策略", "维护措施", "保养方案", "维修方案"],
            "维护策略": ["维护方案", "维护计划", "维护政策", "保养策略", "维修策略"],
            "齿轮箱": ["变速箱", "齿轮传动箱", "传动箱", "齿轮传动装置"],
            "轴承": ["轴承载", "滚动轴承", "滑动轴承", "轴承组件"],
            "振动": ["震动", "振荡", "振动信号", "振动数据"],
            "温度": ["温升", "温度数据", "温度监测", "温度变化"],
            "磨损": ["磨耗", "磨损数据", "磨损监测", "磨损分析"],
            "转速": ["旋转速度", "转动速度", "转速数据", "转数"],
            "负载": ["负荷", "载荷", "负载数据", "承载"],
            "润滑": ["润滑油", "润滑状态", "润滑数据", "润滑监测"]
        }

        # Load custom dictionary into jieba
        self._load_custom_dict()

        print("Semantic matching model loaded successfully")

    def _load_custom_dict(self):
        """Load gearbox operation and maintenance domain custom dictionary into jieba"""
        all_terms = set()
        for main_term, synonyms in self.domain_dict.items():
            all_terms.add(main_term)
            all_terms.update(synonyms)

        for term in all_terms:
            jieba.add_word(term)

    def compute_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        try:
            # Encode texts
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)

            # Calculate cosine similarity
            similarity = 1 - cosine(embeddings[0], embeddings[1])

            # Ensure value is within [0, 1]
            return max(0, min(1, similarity))
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def compute_comprehensive_match(self, task, output_descriptions, aggregation_method='geometric_mean'):
        """
        Calculate comprehensive matching score
        Args:
            task: Task text
            output_descriptions: List of output data descriptions
            aggregation_method: Aggregation method
                - 'geometric_mean': Geometric mean
                - 'harmonic_mean': Harmonic mean
                - 'weighted_mean': Weighted mean
                - 'quadratic_mean': Quadratic mean
                - 'arithmetic_mean': Arithmetic mean
        Returns:
            dict: Dictionary containing comprehensive score and detailed results
        """
        individual_scores = []

        # Calculate similarity between task and each output description
        for description in output_descriptions:
            similarity = self.compute_similarity(task, description)
            individual_scores.append({
                'description': description,
                'similarity': similarity
            })

        # Extract similarity values
        similarities = [item['similarity'] for item in individual_scores]

        if not similarities:
            return {
                'comprehensive_score': 0.0,
                'aggregation_method': aggregation_method,
                'individual_scores': [],
                'task': task,
                'output_descriptions': output_descriptions
            }

        # Get description with highest similarity
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        max_similarity_description = output_descriptions[max_index]

        # Calculate comprehensive score
        if aggregation_method == 'geometric_mean':
            # Geometric mean
            product = np.prod([s + 1e-10 for s in similarities])
            comprehensive_score = product ** (1 / len(similarities))

        elif aggregation_method == 'harmonic_mean':
            # Harmonic mean
            reciprocal_sum = sum(1 / (s + 1e-10) for s in similarities)
            comprehensive_score = len(similarities) / reciprocal_sum

        elif aggregation_method == 'quadratic_mean':
            # Quadratic mean
            square_sum = sum(s ** 2 for s in similarities)
            comprehensive_score = np.sqrt(square_sum / len(similarities))

        elif aggregation_method == 'weighted_mean':
            # Weighted mean (simplified version, all descriptions have equal weight)
            weights = [1.0 / len(similarities)] * len(similarities) if similarities else []
            weighted_sum = sum(w * s for w, s in zip(weights, similarities))
            comprehensive_score = weighted_sum / sum(weights) if weights else 0.0
        elif aggregation_method == 'max':
            comprehensive_score = max_similarity  # Directly use max value
        elif aggregation_method == 'top2_mean':
            # Mean of top 2 highest values
            top2 = sorted(similarities, reverse=True)[:2]
            comprehensive_score = sum(top2) / len(top2) if top2 else 0.0
        elif aggregation_method == 'top3_mean':
            # Mean of top 3 highest values
            top3 = sorted(similarities, reverse=True)[:3]
            comprehensive_score = sum(top3) / len(top3) if top3 else 0.0
        else:  # arithmetic_mean
            # Arithmetic mean
            comprehensive_score = np.mean(similarities)

        return {
            'comprehensive_score': float(comprehensive_score),
            'aggregation_method': aggregation_method,
            'individual_scores': individual_scores,
            'task': task,
            'output_descriptions': output_descriptions,
            'max_similarity_description': max_similarity_description,
            'max_similarity': max_similarity
        }

    def match_task_to_process(self, task, output_descriptions, threshold=0.3):
        """
        Match task to process
        Args:
            task: Task description
            output_descriptions: List of process output data descriptions
            threshold: Matching threshold
        Returns:
            dict: Matching result
        """
        # Calculate comprehensive match
        match_result = self.compute_comprehensive_match(
            task,
            output_descriptions,
            aggregation_method='max'
        )

        # Determine if match
        match_result['is_match'] = match_result['comprehensive_score'] >= threshold

        return match_result


def extract_output_descriptions(model: Dict) -> List[str]:
    """
    Extract output data descriptions from a process model
    Args:
        model: Process model dictionary containing the output_data field
    Returns:
        List[str]: List of output data descriptions
    """
    output_descriptions = []

    # Extract description from output_data
    if "output_data" in model:
        output_data = model.get("output_data", [])
        for item in output_data:
            description = item.get("description", "")
            if description and description.strip():
                output_descriptions.append(description.strip())

    return output_descriptions

def getExpPro(task: str, models: List[Dict]):
    """
    Match the task to each process model
    Args:
        task: Task description
        models: List of process models
    Returns:
        List[Dict]: Matching results for each process, sorted by matching degree
    """
    # Obtain the semantic matcher
    matcher = get_semantic_matcher()

    # print(f"Extracted task: {task}")
    match_results = []

    for model in models:
        # Extract process information
        process_id = model.get("process_id", "")
        process_name = model.get("process_name", "")

        # Extract output data descriptions
        output_descriptions = extract_output_descriptions(model)

        if not output_descriptions:
            continue

        # Calculate matching degree
        match_result = matcher.match_task_to_process(task, output_descriptions)

        # Add process information and original model data
        match_result.update({
            'process_id': process_id,
            'process_name': process_name,
            'output_descriptions': output_descriptions,
            'output_data': model.get("output_data", []),  # Save the complete output data
            'original_model': model  # Save the original model data for later use
        })

        match_results.append(match_result)

    # Sort by comprehensive matching degree (descending)
    match_results.sort(key=lambda x: x['process_id'], reverse=True)

    return match_results