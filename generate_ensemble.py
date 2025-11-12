from src.util.misc import setup_output_and_logging
import logging
import yaml
import argparse
import pandas as pd
import os, json, torch
from src.util.model_util import model_factory
import src.prompt_generation.ensemble_generation as ensemble_generation_util
from typing import List
logger = logging.getLogger(__name__)
#Columns other than "question_id", "variant_id", "generated_variant" that need to be copied from input dataset to output dataset
EXTRA_COLUMNS_PER_DATASET = {"AmbigQA": [],
                                "AmbigInst": ["input"],
                                "TriviaQA": [],
                                "OpenNQ": []} 

    
def generate_zero_shot_clarifications(input_df: pd.DataFrame, config: dict) -> List[dict]:
    dataset = config["dataset"]

    extra_columns = EXTRA_COLUMNS_PER_DATASET[dataset]

    messages = []
    for _, row in input_df.iterrows():
        extra_info = {column: row[column] for column in extra_columns}
        messages.append([{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", 
                         "content": ensemble_generation_util.
                         format_zero_shot_clarification_query(row["question"], dataset, **extra_info)}])
        

    clarification_model = model_factory(config["variation_model"]["config"])
    llm_outputs = clarification_model.generate(messages, 
                                               generation_config=config['variation_model']['generation_config'],
                                               **config['variation_model']['generate_function_parameters'],
                                               **config['variation_model']['generate_function_kwargs'])
    
    clarification_data = []
    for i, row in input_df.iterrows():
        reasoning, clarifications, other_outputs = ensemble_generation_util.parse_zero_shot_clarification_output(llm_outputs[i], dataset)
        if len(clarifications)>0:
            for clarification_idx, clarification in enumerate(clarifications):
                clarification_row= {"question_id": row["question_id"],
                                           "variant_id": clarification_idx,
                                           "generated_variant": clarification}
                clarification_row.update({column_name: row[column_name] for column_name in extra_columns})
                clarification_row.update({"reasoning": reasoning})
                clarification_data.append(clarification_row) 
                
        else: #If the model decides no clarifications are needed
            clarification_row= {"question_id": row["question_id"],
                                           "variant_id": 0,
                                           "generated_variant": row["question"]}
            clarification_row.update({column_name: row[column_name] for column_name in extra_columns})
            clarification_row.update({"reasoning": reasoning})
            clarification_data.append(clarification_row) 
            
        if len(other_outputs)>0:
            logger.debug(f"Question {row['question_id']} model output parsing resulted in the following other outputs: {other_outputs}")
    return clarification_data

def collect_original_questions(input_df: pd.DataFrame, config: dict) -> List[dict]:
    dataset = config["dataset"]

    extra_columns = EXTRA_COLUMNS_PER_DATASET[dataset]
    variant_data = []
    for _, row in input_df.iterrows():
        variant_row = {"question_id": row["question_id"],
                        "variant_id": 0,
                        "generated_variant": row["question"]}
        variant_row.update({column_name: row[column_name] for column_name in extra_columns})
        variant_data.append(variant_row)
    return variant_data

if __name__ == "__main__":
    setup_output_and_logging('experiment_logs/generate_ensemble')
    parser = argparse.ArgumentParser(description="Ensemble generation script.")
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
    input_path = os.path.join(root_path, f"data/eval/{config['dataset']}.json")
    input_df = pd.read_json(input_path)


    if config["ensembling_method"]=="clarification_zeroshot":
        output_data = generate_zero_shot_clarifications(input_df, config)
    elif config["ensembling_method"]=="no_ensembling":
        output_data = collect_original_questions(input_df, config)
    else:
        raise NotImplementedError(f"Ensembling method {config['ensembling_method']} not implemented.")

    variation_model_name=config.get('variation_model', {'model_safename': 'no_model'})['model_safename']
    output_directory = os.path.join(root_path, 
                                    f"data/logs/{config['dataset']}/{config['ensembling_method']}/{variation_model_name}-variations")
    os.makedirs(output_directory, exist_ok=True)
    with open(os.path.join(output_directory, 'question_variants.json'), 'w') as file:
        json.dump(output_data, file, indent=4)

