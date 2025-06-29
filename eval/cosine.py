import json
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize client
client = OpenAI(api_key="", base_url="")

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text string using OpenAI API"""
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        dimensions=1024,
        encoding_format="float"
    )
    embedding = response.data[0].embedding
    return np.array(embedding)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)) * 100)

def evaluate_translation(question: Dict, translation: Dict) -> Dict:
    """Evaluate a single translation"""
    try:
        # Get embeddings
        source_embed = get_embedding(question["sentence"])
        trans_embed = get_embedding(translation["pred_tag"])
        
        # Calculate cosine similarity
        cosine_score = cosine_similarity(source_embed, trans_embed)
        
        return {
            "index": question["index"],
            "sentence": question["sentence"],
            "translation": translation["pred_tag"],
            "cosine": cosine_score
        }
    except Exception as e:
        return {
            "index": question["index"],
            "error": str(e)
        }

def process_batch(batch_items: List[Tuple[Dict, Dict]], executor):
    """Process a batch of translations in parallel"""
    futures = []
    results = []
    
    try:
        # Submit all tasks
        for question, translation in batch_items:
            futures.append(executor.submit(evaluate_translation, question, translation))
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception:
                continue
                
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        raise
    
    return results

def load_data(question_path: str, translation_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load all required data files"""
    with open(question_path, 'r') as f:
        questions = json.load(f)
    
    with open(translation_path, 'r') as f:
        translations = [json.loads(line) for line in f]
    
    return questions, translations

def main():
    batch_size = 20
    max_workers = 20

    #lan = 'zh'
    lan = 'en'
    #type = 'phonic'
    type = 'graphic'

    #strategy = 'vanilla'
    strategy = 'cot'
    #strategy = 'cvo'

    #model = 'qwen-vl-max'
    #model = '4o'
    #model = 'qwq'
    #model = 'o1-mini'
    model = 'deepseek-v3'

    # File paths
    question_path = f"data/textual/{lan}/{type}.json"
    translation_path = f"output/{model}/translation/new/{lan}_{type}_translation_{strategy}.jsonl"
    output_path = f"eval/output/{model}/translation/new/{lan}_{type}_translation_{strategy}_cosine.jsonl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data
    questions, translations = load_data(question_path, translation_path)
    
    # Get items to process
    items_to_process = []
    evaluated_indices = set()
    
    # Load existing results if file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    evaluated_indices.add(result["index"])
        except FileNotFoundError:
            pass  # File doesn't exist yet, which is fine
    
    # Create pairs of items to process
    for question, translation in zip(questions, translations):
        if question["index"] not in evaluated_indices:
            items_to_process.append((question, translation))
            
    if not items_to_process:
        print("No new items to process")
        return
        
    # Process in batches
    batches = [items_to_process[i:i + batch_size] for i in range(0, len(items_to_process), batch_size)]
    
    all_scores = []  # Keep track of scores during processing
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch in tqdm(batches):
                try:
                    batch_results = process_batch(batch, executor)
                    # Write results to jsonl file
                    with open(output_path, 'a') as f:
                        for result in batch_results:
                            if "error" not in result:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                                all_scores.append(result["cosine"])
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
                    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return
        
    # Calculate final statistics including newly processed results
    try:
        with open(output_path, 'r') as f:
            scores = [json.loads(line)["cosine"] for line in f]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"\nFinal average cosine similarity score: {avg_score:.2f}")
    except FileNotFoundError:
        if all_scores:  # Use scores collected during processing if file read fails
            avg_score = sum(all_scores) / len(all_scores)
            print(f"\nFinal average cosine similarity score: {avg_score:.2f}")
        else:
            print("\nNo scores available to calculate average")

if __name__ == "__main__":
    main()
