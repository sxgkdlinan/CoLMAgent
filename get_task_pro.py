import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import jieba
from typing import Dict, List, Any
import config
# 全局匹配器实例
_semantic_matcher = None


def get_semantic_matcher():
    # Get the singleton semantic matcher.
    global _semantic_matcher
    if _semantic_matcher is None:
        _semantic_matcher = GearboxSemanticMatcher()
    return _semantic_matcher


class GearboxSemanticMatcher:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # Initialize the semantic matcher for gearbox operation and maintenance.
        print(f"Load the semantic matching model: {model_name}")
        self.model = SentenceTransformer(config.paraphrase_Url)

        # 齿轮箱运维领域的专用词典
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

        # 加载自定义词典到jieba
        self._load_custom_dict()

        print("The semantic matching model has been loaded successfully.")

    def _load_custom_dict(self):
        # Load the specialized dictionary for gear box operation and maintenance into jieba.
        all_terms = set()
        for main_term, synonyms in self.domain_dict.items():
            all_terms.add(main_term)
            all_terms.update(synonyms)

        for term in all_terms:
            jieba.add_word(term)

    def compute_similarity(self, text1, text2):
        # Calculate the semantic similarity between two texts.
        try:
            # Encoded text
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)

            # Calculate the cosine similarity
            similarity = 1 - cosine(embeddings[0], embeddings[1])

            # Ensure that the value is within the range of [0, 1].
            return max(0, min(1, similarity))
        except Exception as e:
            print(f"计算相似度时出错: {e}")
            return 0.0

    def compute_comprehensive_match(self, task, keywords, aggregation_method='geometric_mean'):
        """
            Calculate the comprehensive matching degree Args:
            task: Task text
            keywords: List of keywords
            aggregation_method: Aggregation method
            - 'geometric_mean': Geometric mean
            - 'harmonic_mean': Harmonic mean
            - 'weighted_mean': Weighted mean
            - 'quadratic_mean': Quadratic mean
            - 'arithmetic_mean': Arithmetic mean Returns:
            dict: Contains comprehensive matching degree and detailed results
        """
        individual_scores = []

        # Calculate the similarity of each keyword.
        for keyword in keywords:
            similarity = self.compute_similarity(task, keyword)
            individual_scores.append({
                'keyword': keyword,
                'similarity': similarity
            })

        # Extract similarity values
        similarities = [item['similarity'] for item in individual_scores]

        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        max_similarity_keyword = keywords[max_index]

        if not similarities:
            return {
                'comprehensive_score': 0.0,
                'aggregation_method': aggregation_method,
                'individual_scores': [],
                'task': task,
                'keywords': keywords
            }

        # Calculate the comprehensive matching degree
        if aggregation_method == 'geometric_mean':
            # geometrical mean
            product = np.prod([s + 1e-10 for s in similarities])
            comprehensive_score = product ** (1 / len(similarities))

        elif aggregation_method == 'harmonic_mean':
            # harmonic mean
            reciprocal_sum = sum(1 / (s + 1e-10) for s in similarities)
            comprehensive_score = len(similarities) / reciprocal_sum

        elif aggregation_method == 'quadratic_mean':
            # quadratic mean
            square_sum = sum(s ** 2 for s in similarities)
            comprehensive_score = np.sqrt(square_sum / len(similarities))

        elif aggregation_method == 'weighted_mean':
            # weighted mean
            weights = self._compute_keyword_weights(keywords)
            weighted_sum = sum(w * s for w, s in zip(weights, similarities))
            comprehensive_score = weighted_sum / sum(weights)
        elif aggregation_method == 'max':
            comprehensive_score = max_similarity  # max
        elif aggregation_method == 'top2_mean':
            # The average of the top two highest values
            top2 = sorted(similarities, reverse=True)[:2]
            comprehensive_score = sum(top2) / len(top2)
        elif aggregation_method == 'top3_mean':
            # The average of the top three highest values
            top3 = sorted(similarities, reverse=True)[:3]
            comprehensive_score = sum(top3) / len(top3)
        else:
            # arithmetic_mean
            comprehensive_score = np.mean(similarities)

        return {
            'comprehensive_score': float(comprehensive_score),
            'aggregation_method': aggregation_method,
            'individual_scores': individual_scores,
            'task': task,
            'keywords': keywords
        }

    def _compute_keyword_weights(self, keywords):
        """计算关键词权重"""
        return [1.0 / len(keywords)] * len(keywords) if keywords else []

    def match_task_to_process(self, task, process_keywords, threshold=0.3):
        """
            Match the tasks to the processes. Args:
            Task: Task description
            Process Keywords: List of process keywords
            Threshold: Matching threshold Returns:
            dict: Matching result
        """
        # Calculate Comprehensive Matching Degree
        match_result = self.compute_comprehensive_match(
            task,
            process_keywords,
            aggregation_method='top2_mean'
        )

        # Determine whether it matches.
        match_result['is_match'] = match_result['comprehensive_score'] >= threshold

        return match_result


def getTaskPro(task: str, models: List[Dict]):
    """
        Match the tasks to each process model. Args:
        Task: Task description
        Models: List of process models Returns:
        List[Dict]: Matching results for each process, sorted by matching degree
    """
    # Get the semantic matcher
    matcher = get_semantic_matcher()

    match_results = []

    for model in models:
        process_id = model.get("process_id", "")
        process_name = model.get("process_name", "")
        process_keywords = model.get("process_keywords", [])

        if not process_keywords:
            continue

        # Calculate the matching degree
        match_result = matcher.match_task_to_process(task, process_keywords)

        # Add process information
        match_result.update({
            'process_id': process_id,
            'process_name': process_name,
            'process_keywords': process_keywords
        })

        match_results.append(match_result)

    # Sort by comprehensive matching degree (in descending order)
    match_results.sort(key=lambda x: x['process_id'], reverse=True)

    return match_results