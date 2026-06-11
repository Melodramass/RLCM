from rlcm.scripts.data.uncertainty_deepscaler_dataset import extract_confidence
from verl.utils.reward_score.math import compute_score as math_compute_score

def compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, **kwargs):
    confidence = extract_confidence(solution_str)
    math_score = math_compute_score(solution_str, ground_truth)
    return math_score - (math_score - confidence)^2
