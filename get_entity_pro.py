import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import jieba
from typing import List, Dict, Any, Optional, Tuple
import config

_object_matcher = None
def get_object_matcher():
    global _object_matcher
    if _object_matcher is None:
        _object_matcher = GearboxObjectMatcher()
    return _object_matcher


class GearboxObjectMatcher:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize the gearbox object matcher
        """
        print(f"Loading semantic matching model: {model_name}")
        self.model = SentenceTransformer(config.paraphrase_Url)

        # Gearbox object type dictionary (for semantic matching)
        self.object_type_dict = {
            "齿轮箱": ["变速箱", "传动箱", "齿轮传动装置", "gearbox", "齿轮传动箱"],
            "齿轮": ["齿轮", "齿轮件", "齿圈", "gear", "内圈齿轮", "外圈齿轮", "行星齿轮"],
            "轴承": ["轴承", "滚动轴承", "滑动轴承", "轴承组件", "bearing", "轴承载"],
            "轴": ["轴", "传动轴", "主轴", "转轴", "shaft"],
            "联轴器": ["联轴器", "耦合器", "coupling"],
            "密封件": ["密封", "密封件", "油封", "密封圈", "seal"],
            "润滑油": ["润滑油", "润滑脂", "润滑剂", "lubricant", "机油"],
        }

        # Load custom dictionary into jieba
        self._load_custom_dict()

        print("Object matching model loaded successfully")

    def _load_custom_dict(self):
        """Load gearbox object dictionary into jieba"""
        all_terms = set()
        for main_term, synonyms in self.object_type_dict.items():
            all_terms.add(main_term)
            all_terms.update(synonyms)

        for term in all_terms:
            jieba.add_word(term)

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            if not text1 or not text2:
                return 0.0

            # Encode texts
            embeddings = self.model.encode([text1, text2], convert_to_numpy=True)

            # Calculate cosine similarity
            similarity = 1 - cosine(embeddings[0], embeddings[1])

            # Ensure value is within [0, 1]
            return max(0, min(1, similarity))
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0

    def match_objects_direct(self,
                             user_device_name: Optional[str],
                             user_device_id: Optional[str],
                             process_device_names: List[str],
                             process_device_ids: List[str]) -> Dict[str, Any]:
        """
        Directly match object name and ID (using pre-extracted device_name and device_id)
        Args:
            user_device_name: Device name provided by user
            user_device_id: Device ID provided by user
            process_device_names: List of device names in the process
            process_device_ids: List of device IDs in the process
        Returns:
            dict: Matching result
        """
        # Initialize result
        result = {
            'exact_name_match': False,
            'exact_id_match': False,
            'semantic_name_match': False,
            'semantic_id_match': False,
            'name_similarity': 0.0,
            'id_similarity': 0.0,
            'combined_score': 0.0,
            'matched_name': None,
            'matched_id': None,
            'match_strategy': None
        }

        # Strategy 1: Exact ID match (highest priority)
        if user_device_id and process_device_ids:
            # Handle possible case issues
            user_id_lower = user_device_id.lower().strip()
            process_ids_lower = [str(pid).lower().strip() for pid in process_device_ids if pid]

            if user_id_lower in process_ids_lower:
                idx = process_ids_lower.index(user_id_lower)
                result['exact_id_match'] = True
                result['matched_id'] = process_device_ids[idx]
                result['id_similarity'] = 1.0
                result['match_strategy'] = 'exact_id_match'

        # Strategy 2: Exact name match (second priority)
        if not result['exact_id_match'] and user_device_name and process_device_names:
            # Handle possible case and whitespace issues
            user_name_clean = user_device_name.strip()
            process_names_clean = [str(name).strip() for name in process_device_names if name]

            # Direct string comparison
            if user_name_clean in process_names_clean:
                idx = process_names_clean.index(user_name_clean)
                result['exact_name_match'] = True
                result['matched_name'] = process_device_names[idx]
                result['name_similarity'] = 1.0
                result['match_strategy'] = 'exact_name_match'
            # Check for containment relationship
            else:
                for i, process_name in enumerate(process_names_clean):
                    if user_name_clean in process_name or process_name in user_name_clean:
                        result['exact_name_match'] = True
                        result['matched_name'] = process_device_names[i]
                        result['name_similarity'] = 0.9  # Partial match score
                        result['match_strategy'] = 'partial_name_match'
                        break

        # Strategy 3: Semantic name match
        if not result['exact_id_match'] and not result['exact_name_match']:
            if user_device_name and process_device_names:
                # Calculate semantic similarity with all process device names
                similarities = []
                for process_name in process_device_names:
                    if process_name and user_device_name:
                        similarity = self.compute_semantic_similarity(
                            user_device_name, str(process_name)
                        )
                        similarities.append((process_name, similarity))

                if similarities:
                    # Find the highest similarity
                    best_match = max(similarities, key=lambda x: x[1])
                    if best_match[1] > 0.7:  # Set semantic matching threshold
                        result['semantic_name_match'] = True
                        result['matched_name'] = best_match[0]
                        result['name_similarity'] = best_match[1]
                        result['match_strategy'] = 'semantic_name_match'

        # Strategy 4: ID pattern match (partial match)
        if not result['exact_id_match'] and user_device_id and process_device_ids:
            # Check if the process ID contains the user-provided ID
            user_id_clean = str(user_device_id).strip()
            for process_id in process_device_ids:
                process_id_str = str(process_id).strip()
                if process_id_str and user_id_clean:
                    if user_id_clean in process_id_str or process_id_str in user_id_clean:
                        result['semantic_id_match'] = True
                        result['matched_id'] = process_id
                        result['id_similarity'] = 0.8  # Partial match similarity
                        result['match_strategy'] = 'partial_id_match'
                        break

        # Calculate combined score
        if result['exact_id_match']:
            result['combined_score'] = 1.0
        elif result['exact_name_match'] and result['match_strategy'] == 'exact_name_match':
            result['combined_score'] = 0.9
        elif result['exact_name_match'] and result['match_strategy'] == 'partial_name_match':
            result['combined_score'] = 0.85
        elif result['semantic_name_match'] and result['name_similarity'] > 0.8:
            result['combined_score'] = 0.8
        elif result['semantic_id_match']:
            result['combined_score'] = 0.7
        elif result['semantic_name_match']:
            result['combined_score'] = result['name_similarity'] * 0.6
        else:
            # If no match, try using the average of semantic similarities
            scores = []
            if result['name_similarity'] > 0:
                scores.append(result['name_similarity'])
            if result['id_similarity'] > 0:
                scores.append(result['id_similarity'])
            result['combined_score'] = np.mean(scores) if scores else 0.0

        return result

    def match_object_to_process_direct(self,
                                       user_device_name: Optional[str],
                                       user_device_id: Optional[str],
                                       process_device_names: List[str],
                                       process_device_ids: List[str],
                                       threshold: float = 0.5) -> Dict[str, Any]:
        """
        Directly match user input object to a process (using pre-extracted device_name and device_id)
        Args:
            user_device_name: User device name
            user_device_id: User device ID
            process_device_names: List of device names in the process
            process_device_ids: List of device IDs in the process
            threshold: Matching threshold
        Returns:
            dict: Matching result
        """
        # Perform object matching
        match_result = self.match_objects_direct(
            user_device_name,
            user_device_id,
            process_device_names,
            process_device_ids
        )

        # Add user-provided object information
        match_result['user_device_name'] = user_device_name
        match_result['user_device_id'] = user_device_id

        # Determine if match
        match_result['is_match'] = match_result['combined_score'] >= threshold

        # Generate match description
        if match_result['exact_id_match']:
            match_result['match_description'] = f"Exact ID match: {match_result['matched_id']}"
        elif match_result['exact_name_match'] and match_result['match_strategy'] == 'exact_name_match':
            match_result['match_description'] = f"Exact name match: {match_result['matched_name']}"
        elif match_result['exact_name_match'] and match_result['match_strategy'] == 'partial_name_match':
            match_result['match_description'] = f"Partial name match: {match_result['matched_name']}"
        elif match_result['semantic_name_match']:
            match_result[
                'match_description'] = f"Semantic name match: {match_result['matched_name']} (similarity: {match_result['name_similarity']:.2f})"
        elif match_result['semantic_id_match']:
            match_result['match_description'] = f"Partial ID match: {match_result['matched_id']}"
        else:
            match_result['match_description'] = "No object matched"

        return match_result

def extract_process_devices(model: Dict) -> Tuple[List[str], List[str]]:
    """
    Extract device names and IDs from a process model
    Args:
        model: Process model dictionary
    Returns:
        Tuple[List[str], List[str]]: (list of device names, list of device IDs)
    """
    device_names = model.get("device_name", [])
    device_ids = model.get("device_id", [])

    # Ensure return values are lists
    if not isinstance(device_names, list):
        device_names = [device_names] if device_names else []
    if not isinstance(device_ids, list):
        device_ids = [device_ids] if device_ids else []

    # Filter empty values and convert to strings
    device_names = [str(name).strip() for name in device_names if name is not None and str(name).strip()]
    device_ids = [str(id_).strip() for id_ in device_ids if id_ is not None and str(id_).strip()]

    return device_names, device_ids


def getEntityPro(user_device_name: Optional[str],
                                user_device_id: Optional[str],
                                models: List[Dict]) -> List[Dict]:
    """
    Directly match user input object to each process model (using pre-extracted device_name and device_id)
    Args:
        user_device_name: User device name
        user_device_id: User device ID
        models: List of process models
    Returns:
        List[Dict]: Matching result for each process, sorted by matching score
    """
    # Get the object matcher
    matcher = get_object_matcher()

    match_results = []

    for model in models:
        # Extract process information
        process_id = model.get("process_id", "")
        process_name = model.get("process_name", "")
        device_names = model.get("device_name", [])
        device_ids = model.get("device_id", [])
        # Extract device information
        #device_names, device_ids = extract_process_devices(model)

        # Calculate matching score
        match_result = matcher.match_object_to_process_direct(
            user_device_name,
            user_device_id,
            device_names,
            device_ids,
            threshold=0.5
        )

        # Add process information
        match_result.update({
            'process_id': process_id,
            'process_name': process_name,
            'process_device_names': device_names,
            'process_device_ids': device_ids,
            'original_model': model
        })

        match_results.append(match_result)

    # Sort by comprehensive matching score (descending)
    match_results.sort(key=lambda x: x['process_id'], reverse=True)

    return match_results
