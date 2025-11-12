from tqdm import tqdm
from src.uncertainty_metrics.common import compute_discrete_semantic_entropy, compute_kernel_language_entropy
from src.uncertainty_metrics.common import compute_disagreement_measure
from sentence_transformers import SentenceTransformer
from src.uncertainty_metrics.semantic_entropy import EntailmentDeberta
import torch, yaml, argparse, os, logging, json
import pandas as pd
import numpy as np
from typing import List
from src.util.misc import setup_output_and_logging
logger = logging.getLogger(__name__)


UNCERTAINTY_METRICS = {"discrete_semantic_entropy": compute_discrete_semantic_entropy,
                        "kernel_language_entropy": compute_kernel_language_entropy}

def compute_uncertainty_values(input_df: pd.DataFrame, metric: str, config: dict) -> List[dict]:
    assert metric in UNCERTAINTY_METRICS, f"Uncertainty metric should be one of {UNCERTAINTY_METRICS.keys()}"
    print(f"Computing uncertainty metric: {metric}")
    compute_function = UNCERTAINTY_METRICS[metric]
    model = EntailmentDeberta()
    output_data = []
    question_ids = np.unique(input_df["question_id"]) #Sorted unique values
    for question_id in tqdm(question_ids):
        question_df = input_df[input_df["question_id"] == question_id]
        if config["ensembling_method"] == "no_ensembling":
            answers = list(question_df["generated_answer"])
            uncertainty = compute_function(answers, model)
            output_data.append({"question_id": int(question_id),
                                f"{metric}_total": uncertainty})
        else:
            answers = list(question_df.groupby("variant_id")['generated_answer'].agg(list))
            total, avg, disagreement = compute_disagreement_measure(answers, model, compute_function)
            output_data.append({"question_id": int(question_id),
                                f"{metric}_total": total,
                                f"{metric}_avg": avg,
                                f"{metric}_disagreement": disagreement})
    return output_data

# First parse config
if __name__=='__main__':
    setup_output_and_logging('experiment_logs/compute_uncertainty')
    parser = argparse.ArgumentParser(description="Compute uncertainty without embeddings script.")
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True, 
        help="Path to the config file")
    args = parser.parse_args()
    logger.info(f"Using config file: {args.config_path}")
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Using config:\n{yaml.dump(config)}")
    root_path = os.path.dirname(__file__)
    variation_model_name=config.get('variation_model', {'model_safename': 'no_model'})['model_safename']
    input_directory = os.path.join(root_path, 
                                    f"data/logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations")
    target_model_name=config['target_model']['model_safename']
    input_path = os.path.join(input_directory, f'{target_model_name}-answers.json')
    input_df = pd.read_json(input_path)
    
    if config["uncertainty_metric"] == "all":
        output_data = [dict() for _ in range(len(np.unique(input_df["question_id"])))]
        for metric in UNCERTAINTY_METRICS.keys():
            metric_output = compute_uncertainty_values(input_df, metric, config)
            for old_dict, new_dict in zip(output_data, metric_output):
                old_dict.update(new_dict)
    else:
        output_data = compute_uncertainty_values(input_df, config["uncertainty_metric"], config)

    output_directory = input_directory
    with open(os.path.join(output_directory, f'{target_model_name}-uncertainties_wo_embeddings.json'), 'w') as file:
        json.dump(output_data, file, indent=4)

