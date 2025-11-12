import os, re
from typing import List, Tuple

def format_zero_shot_clarification_query(question: str, dataset: str, **kwargs) -> str:

    template_path = os.path.join(os.path.dirname(__file__), f'../../prompt_templates/{dataset}/zero_shot_clarification.txt')
    with open(template_path, 'r') as file:
        template = file.read()

    if dataset=="AmbigQA":
        prompt_full = template + "\n\n" + "**Question to Analyse**"
        prompt_full = prompt_full + f"\n### Question:\n{question}"
        return prompt_full
    if dataset=="AmbigInst":
        prompt_full = template + "\n\n" + "**Task Description and Input**"
        prompt_full = prompt_full + f"\n### Task description:\n{question}"
        prompt_full = prompt_full + f"\n\n### Task input:\n{kwargs['input']}"
        return prompt_full
    if dataset=="TriviaQA":
        prompt_full = template + "**Task Input**"
        prompt_full = prompt_full + f"\n### Original Question:\n{question}"
        return prompt_full
    if dataset=="OpenNQ":
        prompt_full = template + "**Task Input**"
        prompt_full = prompt_full + f"\n### Original Question:\n{question}"
        return prompt_full
    raise NotImplementedError(f"Zero shot clarification query formatting is not implemented for dataset {dataset}")


def parse_zero_shot_clarification_output(model_answer: str, dataset: str) -> Tuple[str, List[str], List[str]]:
    if dataset in ["AmbigQA", "AmbigInst"]:
        reasoning = ""
        clarifications = []
        other_outputs = []

        # Extract the Analyses section
        analyses_match = re.search(r"### Analyses:\s*(.*?)(?=### Clarifications)", model_answer, re.DOTALL)
        if analyses_match:
            reasoning = analyses_match.group(1).strip()

        # Extract the Clarifications section
        clarifications_match = re.search(r"### Clarifications:\s*(.*)", model_answer, re.DOTALL)
        if clarifications_match:
            clarifications_text = clarifications_match.group(1).strip()
            if "No clarification needed".lower() in clarifications_text.lower():
                clarifications = []
            else:
                clarifications = re.findall(r"#\d+\s+(.*?)(?=(?:\-\-\-)|#|\Z)", clarifications_text, re.DOTALL)
                clarifications = [clarification.strip() for clarification in clarifications]

        # Extract anything else that doesn't match either section (extra/hallucinated content)
        expected_sections = re.findall(r"(### Analyses:.*?)(?=### Clarifications)", model_answer, re.DOTALL)
        expected_sections += re.findall(r"(### Clarifications:.*)", model_answer, re.DOTALL)
        combined_expected = "\n".join(expected_sections)
        
        extra_lines = [line for line in model_answer.strip().splitlines()
                    if line.strip() and line not in combined_expected]
        
        other_outputs = [line.strip() for line in extra_lines]

        return reasoning, clarifications, other_outputs
    if dataset in ["TriviaQA", "OpenNQ"]:
        reasoning = ""
        clarifications = []
        other_outputs = []

        # Extract the Clarifications section
        clarifications_match = re.search(r"### Rephrasings:\s*(.*)", model_answer, re.DOTALL)
        if clarifications_match:
            clarifications_text = clarifications_match.group(1).strip()
            clarifications = re.findall(r"#\d+\s+(.*?)(?=(?:\-\-\-)|#|\Z)", clarifications_text, re.DOTALL)
            clarifications = [clarification.strip() for clarification in clarifications]

        # Extract anything else that doesn't match that section (extra/hallucinated content)
        expected_sections = re.findall(r"(### Rephrasings:.*)", model_answer, re.DOTALL)
        combined_expected = "\n".join(expected_sections)
        
        extra_lines = [line for line in model_answer.strip().splitlines()
                    if line.strip() and line not in combined_expected]
        
        other_outputs = [line.strip() for line in extra_lines]

        return reasoning, clarifications, other_outputs
    raise NotImplementedError(f"Zero shot clarification response parsing is not implemented for dataset {dataset}")

