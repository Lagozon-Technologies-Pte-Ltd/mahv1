import json

from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings


def get_example_selector(json_file_path: str):
    """
    Returns a SemanticSimilarityExampleSelector initialized with examples from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing examples
        
    Returns:
        SemanticSimilarityExampleSelector: Selector configured with examples
    """
    # Load examples from JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        examples = json.load(file)
    
    # Validate examples structure
    if not isinstance(examples, list):
        raise ValueError("JSON file should contain a list of examples")
    if len(examples) == 0:
        raise ValueError("No examples found in JSON file")
    if not all(isinstance(example, dict) and 'input' in example and 'query' in example for example in examples):
        raise ValueError("Each example should be a dictionary with 'input' and 'query' keys")
    
    # Create example selector
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        Chroma,
        k=3,
        input_keys=["input"],
    )
    
    return example_selector


