import os
import re 
import logging
logger = logging.getLogger(__name__)

def format_query(question: str, dataset: str, **kwargs) -> str:
    template_path = os.path.join(os.path.dirname(__file__), f'../../prompt_templates/{dataset}/generate_answer.txt')
    with open(template_path, 'r') as file:
        template = file.read()

    if dataset=="AmbigQA":
        prompt_full = template + question
        return prompt_full
    if dataset=="AmbigInst":
        prompt_full = template + f"Task description: {question}\nInput: {kwargs['input']}"
        return prompt_full
    if dataset=="TriviaQA":
        prompt_full = template + f"Question:\nQ: {question}"
        return prompt_full
    if dataset=="OpenNQ":
        prompt_full = template + f"Question:\nQ: {question}"
        return prompt_full
    raise NotImplementedError(f"Answer generation query formatting is not implemented for dataset {dataset}")


def parse_generated_answer(llm_output: str, dataset: str) -> str:
    if dataset=="AmbigQA":
        match = re.search(r"Answer:\s*(.*)", llm_output, re.DOTALL)
        if match: 
            return match.group(1).strip()
        else: 
            if "Answer:".lower() in llm_output.lower():
                return ''
            else:
                return llm_output
    if dataset=="AmbigInst":
        pattern = r"Reasoning:\s*(.*?)\s*Answer:\s*(.*)"
        match = re.search(pattern, llm_output, re.DOTALL)

        if not match:
            pattern = r"Answer:\s*(.*)"
            match = re.search(pattern, llm_output, re.DOTALL)
            if not match:
                pattern = r"Answer(.*)"
                match = re.search(pattern, llm_output, re.DOTALL)
                if not match:
                    return ''
                
            return match.group(1).strip()
            
        answer = match.group(2).strip()
        return answer
    
    if dataset in ["TriviaQA", "OpenNQ"]:
        answer = llm_output
        patterns = [
            re.compile(r'The question is unclear\. Random guess:\s*', re.IGNORECASE),
            re.compile(r'Random guess:\s*', re.IGNORECASE),
            re.compile(r'^A:\s*', re.IGNORECASE)
        ]
        #Try flexible patterns (anywhere in the string)
        for pattern in patterns:
            match = pattern.search(answer)
            if match:
                answer = answer[match.end():].strip()
                break

        # Remove surrounding quotes
        answer = answer.strip('"“”')
         # Remove trailing punctuation
        answer = re.sub(r'[.?!]+$', '', answer)
        return answer

    raise NotImplementedError(f"Parsing model generated answer is not implemented for dataset {dataset}")