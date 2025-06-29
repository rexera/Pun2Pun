import json
import os
from typing import List, Dict
from tqdm import tqdm
from textwrap import dedent
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize OpenAI client
client = OpenAI(api_key="", base_url="")
MODEL = "qwen2.5-7b-instruct"

PROMPT =  '''
You are a helpful assistant that determines if the model prediction covers the annotation.
You should output only 'yes' if it does, and 'no' if not.
Focus only on content and semantics, ignore the style. Minor differences or extended explanations are acceptable if it does hit the annotation.

--------------------------------
The original question for the model is:
这句话中双关语的机制是什么？
--------------------------------
'''


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def ensure_dir(file_path):
    """Ensure output directory exists"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def compare_puns_with_llm(pred: str, anno: str, index: int) -> Dict:
    """Compare prediction with annotation using LLM."""
    
    user_prompt = f"""Please compare these two answers:

Prediction: {pred}
Annotation: {anno}

Does the prediction cover the annotation? Output only 'yes' or 'no'."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": dedent(PROMPT)},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().lower()
        return {
            'index': index,
            'pred': pred,
            'anno': anno,
            'is_correct': result == 'yes'
        }
    except Exception as e:
        print(f"Error in LLM API call for index {index}: {e}")
        return {
            'index': index,
            'pred': pred,
            'anno': anno,
            'is_correct': False,
            'error': str(e)
        }

def process_batch(batch_items, executor):
    """Process a batch of data"""
    futures = []
    results = []
    
    try:
        for item in batch_items:
            futures.append(
                executor.submit(
                    compare_puns_with_llm,
                    item['pred'],
                    item['anno'],
                    item['index']
                )
            )
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing future: {str(e)}")
                continue
    except KeyboardInterrupt:
        print("\nDetected keyboard interrupt, cancelling pending tasks...")
        for future in futures:
            future.cancel()
        raise
    
    return results

def write_jsonl_file(file_path, data):
    """Write data to jsonl file"""
    ensure_dir(file_path)
    with open(file_path, 'a', encoding='utf-8') as file:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(f"{json_line}\n")

def evaluate_puns(pred_file: str, true_file: str, output_file: str, batch_size: int = 10, max_workers: int = 5) -> Dict:
    """Evaluate predicted puns against annotations."""
    
    # Load predictions and annotations
    pred_data = load_jsonl(pred_file)
    anno_data = load_jsonl(true_file)
    
    # Create index-based dictionaries
    pred_dict = {item.get('index', i): item.get('pred_tag', '') for i, item in enumerate(pred_data)}  # Try 'response' instead of 'pun'
    anno_dict = {item.get('index', i): item.get('explanation', '') for i, item in enumerate(anno_data)}
    
    # Get indices from both datasets
    pred_indices = set(pred_dict.keys())
    anno_indices = set(anno_dict.keys())
    common_indices = sorted(pred_indices & anno_indices)
    
    # Print index statistics
    print(f"\nIndex Statistics:")
    print(f"Prediction indices: {len(pred_indices)}")
    print(f"Annotation indices: {len(anno_indices)}")
    print(f"Common indices: {len(common_indices)}")
    
    if not common_indices:
        raise ValueError("No common indices found between prediction and annotation files")
    
    # Prepare data for batch processing
    evaluation_data = []
    for idx in common_indices:
        pred = pred_dict[idx]
        anno = anno_dict[idx]
        
        if not pred or not anno:
            print(f"Empty value found for index {idx}: pred='{pred}', anno='{anno}' - skipping")
            continue
            
        evaluation_data.append({
            'pred': pred,
            'anno': anno,
            'index': idx
        })
    
    print(f"\nEvaluation Setup:")
    print(f"Total valid pairs for evaluation: {len(evaluation_data)}")
    
    # Split data into batches
    batches = [evaluation_data[i:i + batch_size] for i in range(0, len(evaluation_data), batch_size)]
    
    total_correct = 0
    all_results = []
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_num, batch in enumerate(tqdm(batches), 1):
                try:
                    # Process current batch
                    results = process_batch(batch, executor)
                    results.sort(key=lambda x: x['index'])
                    
                    # Update statistics
                    total_correct += sum(1 for r in results if r['is_correct'])
                    all_results.extend(results)
                    
                    # Write batch results
                    write_jsonl_file(output_file, results)
                    
                except Exception as e:
                    print(f"Error processing batch {batch_num}: {str(e)}")
                    continue
    
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Partial results have been saved.")
    
    total = len(evaluation_data)
    
    results = {
        'total_pred_indices': len(pred_indices),
        'total_true_indices': len(anno_indices),
        'common_indices': len(common_indices),
        'evaluated_pairs': total,
        'correct': total_correct,
        'accuracy': total_correct/total if total > 0 else 0
    }
    
    return results

def main():
    
    #lan = 'zh'
    lan = 'en'
    #type = 'phonic'
    type = 'graphic'

    #strategy = 'vanilla'
    strategy = 'cot'
    #strategy = 'cvo'

    model = 'qwen-vl-max'
    #model = '4o'
    #model = 'qwq'
    #model = 'o1-mini'

    # File paths
    pred_file = f"output/{model}/{lan}_{type}_explanation_pred_{strategy}.jsonl"
    true_file = f"data/textual/{lan}/{lan}_{type}_explanation.jsonl"
    output_file = f"eval/output/{model}/{lan}_{type}_explanation_{strategy}.jsonl"
    
    # Ensure output directory exists
    ensure_dir(output_file)
    
    # Run evaluation
    results = evaluate_puns(
        pred_file=pred_file,
        true_file=true_file,
        output_file=output_file,
        batch_size=20,
        max_workers=20
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total prediction indices: {results['total_pred_indices']}")
    print(f"Total annotation indices: {results['total_true_indices']}")
    print(f"Common indices: {results['common_indices']}")
    print(f"Evaluated pairs: {results['evaluated_pairs']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
