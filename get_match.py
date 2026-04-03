import get_expectation_pro as ep
import get_task_pro as tp
from typing import Dict, List
import get_entity_pro as etp
import joblib
import config

# Load XGBoost
model = joblib.load(config.XGBoost_Url)
def integrate_scores(task_probability, exp_probability, etp_probability):
    """
    Combine three probability results into a specified format
    Args:
        task_probability: Task probability list
        exp_probability: Output data probability list
        etp_probability: Entity probability list
    Returns:
        list: Organized result list
    """
    results = []

    # Ensure all three lists have the same length
    if len(task_probability) != len(exp_probability) or len(task_probability) != len(etp_probability):
        print("Warning: the three probability lists have different lengths")
        return results

    for i in range(len(task_probability)):
        # Extract information for each process
        process_info = [
            task_probability[i]['process_id'],
            task_probability[i]['process_name'],
            task_probability[i]['comprehensive_score'],
            exp_probability[i]['comprehensive_score'],
            etp_probability[i]['combined_score']
        ]
        results.append(process_info)

    return results


def get_match(task: str, device_name: str, device_id: str, output_data: str, models: List[Dict],
              weights: List[float] = [0.5, 0.3, 0.2],
              threshold: float = 0.5) -> List[Dict]:
    # Task matching degree
    task_probability = tp.getTaskPro(task, models)
    # Expected matching degree
    exp_probability = ep.getExpPro(output_data, models)
    # Entity matching degree
    etp_probability = etp.getEntityPro(device_name, device_id, models)


    integrated_results = integrate_scores(task_probability, exp_probability, etp_probability)

    # Optimized code
    features = [result[2:5] for result in integrated_results]  # using list comprehension

    # Get prediction probabilities (if the model supports it)
    probabilities = model.predict_proba(features)
    # Extract probability of the positive class (class 1) as confidence score
    confidence_scores = [prob[1] for prob in probabilities]  # assuming class 1 is the positive class
    # Find the maximum confidence and its index using built-in functions
    max_confidence = max(confidence_scores)
    max_index = confidence_scores.index(max_confidence)
    best_process_name = integrated_results[max_index][1]
    best_process_id = integrated_results[max_index][0]
    # Output the process with the highest confidence
    print(f"\n Highest confidence process: {best_process_id} - {best_process_name} (confidence: {max_confidence:.3f})")

    return best_process_id, best_process_name, max_confidence

