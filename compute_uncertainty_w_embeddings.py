from tqdm import tqdm
import pickle
from src.uncertainty_metrics.common import compute_von_neuman_entropy_no_model, compute_predictive_kernel_entropy_no_model
from src.uncertainty_metrics.common import compute_disagreement_measure_no_model
from src.util.embeddings_util import embedding_factory
import torch, yaml, argparse, os, logging, json
import pandas as pd
import numpy as np
from typing import List
from src.util.misc import setup_output_and_logging
logger = logging.getLogger(__name__)


UNCERTAINTY_METRICS = {"predictive_kernel_entropy": compute_predictive_kernel_entropy_no_model, 
                       "von_neuman_entropy": compute_von_neuman_entropy_no_model}
def compute_embeddings(input_df: pd.DataFrame, config: dict) -> List[np.ndarray]:
    embedding_model = embedding_factory(config['embedding_model'])
    answers = list(input_df["generated_answer"])
    embeddings = []
    print("Generating embeddings..")
    for i in tqdm(range(0, len(answers), config['embedding_model']['batch_size'])):
        batch = answers[i:i+config['embedding_model']['batch_size']]
        batch_embeddings = embedding_model.encode(batch, config['embedding_model']['encoding_arguments'])
        embeddings = embeddings + list(batch_embeddings)
    return embeddings

def compute_uncertainty_values(input_df: pd.DataFrame, metric: str, config: dict) -> List[dict]:
    assert metric in UNCERTAINTY_METRICS, f"Uncertainty metric should be one of {UNCERTAINTY_METRICS.keys()}"
    print(f"Computing uncertainty metric: {metric}")
    compute_function = UNCERTAINTY_METRICS[metric]
    output_data = []
    question_ids = np.unique(input_df["question_id"]) #Sorted unique values
    for question_id in tqdm(question_ids):
        question_df = input_df[input_df["question_id"] == question_id]
        if config["ensembling_method"] == "no_ensembling":
            embeddings = list(question_df["embedding"])
            uncertainty = compute_function(embeddings, gamma=config['kernel_gamma']) #TODO TEST
            output_data.append({"question_id": int(question_id),
                                f"{metric}_total": uncertainty})
        else:
            embeddings = list(question_df.groupby("variant_id")['embedding'].agg(list))
            total, avg, disagreement = compute_disagreement_measure_no_model(embeddings, compute_function, gamma=config['kernel_gamma']) #TODO TEST
            output_data.append({"question_id": int(question_id),
                                f"{metric}_total": total,
                                f"{metric}_avg": avg,
                                f"{metric}_disagreement": disagreement})
    return output_data

# First parse config
if __name__=='__main__':
    setup_output_and_logging('experiment_logs/compute_uncertainty')
    parser = argparse.ArgumentParser(description="Compute uncertainty with embeddings script.")
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
    
    #First we compute the embeddings
    embeddings = compute_embeddings(input_df, config)
    input_df["embedding"] = embeddings

    #Then compute the metrics
    if config["uncertainty_metric"] == "all":
        output_data = [dict() for _ in range(len(np.unique(input_df["question_id"])))]
        for metric in UNCERTAINTY_METRICS.keys():
            metric_output = compute_uncertainty_values(input_df, metric, config)
            for old_dict, new_dict in zip(output_data, metric_output):
                old_dict.update(new_dict)
    else:
        output_data = compute_uncertainty_values(input_df, config["uncertainty_metric"], config)

    output_directory = input_directory
    embedding_model_safename = config['embedding_model']['model_safename']
    with open(os.path.join(output_directory, f'{target_model_name}-uncertainties_w_embeddings-{embedding_model_safename}.json'), 'w') as file:
        json.dump(output_data, file, indent=4)
    with open(os.path.join(output_directory, f'{target_model_name}-embeddings-{embedding_model_safename}.pkl'), "wb") as f:
        pickle.dump(embeddings, f)

