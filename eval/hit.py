import json
import os
from typing import List, Dict
from tqdm import tqdm
from textwrap import dedent
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

definition = '''同音双关（homophonic pun）指两个词发音相同或相似，但意义不同，由此构成双关；
同形双关（homographic pun）指同一个词能够被理解为两重意思，或两个词形相同或相似，但意义不同，由此构成双关。
'''

# Initialize OpenAI client
client = OpenAI(api_key="", base_url="")
MODEL = "qwen-max-latest"

PROMPT_zh = f'''You are a translation expert and native English speaker, responsible for determining whether the model output contains valid puns and evaluating their appropriateness and fluency in English. Please be strict and ensure accurate judgment.

The model's task is to translate Chinese puns into English puns. Your task is to determine if the given translation is valid.

The definition of puns is as follows:
{definition}

For homophonic puns, the translation must contain words with the same or similar pronunciation but different meanings.
For homographic puns, the translation must contain words with the same or similar form but different meanings.

You will be given the original Chinese sentence and its translation. You need to judge according to the following steps:

1. **Check Translation Fluency**:
   - Determine if the translation follows English grammar structure and flows naturally. If the translation is unnatural or doesn't conform to English language conventions, immediately answer "No" and briefly explain the issues.

2. **Determine if a Pun Exists**:
   - For homophonic puns, are there words with same/similar pronunciation but different meanings? If the pronunciation difference is too large, answer "No" directly.
   - For homographic puns, are there words with same/similar form but different meanings?

3. **Analyze Pun Appropriateness**:
   - If a pun exists in the translation, analyze whether it's appropriate and can be naturally understood in English.
   - For homophonic puns, explain the words with similar/same pronunciation and their different meanings.
   - For homographic puns, explain the words with similar form and their different meanings.

4. **Cultural and Contextual Considerations**:
   - Ensure your judgment considers native English speakers' comprehension and acceptance. If the pun is unnatural or fails to create effective humor or double meaning in English, answer "No".
   - We allow translating a source language homophonic pun into a homographic pun, or a source language homographic pun into a homophonic pun.
   - We do not allow using parenthetical annotations to convey the original pun's meaning, nor directly translating both meanings from the source language.

Final Answer: Yes/No
'''

PROMPT_en = f'''你是一位翻译专家，中文母语者，负责判断模型输出是否包含有效的双关语。要求严格，确保判断准确。

模型的任务是将英文中的双关语翻译成中文双关语。你的任务是判断给定的翻译是否有效。

双关语的定义如下：
{definition}

你会得到原始英文句子和翻译句子。你需要按照以下步骤判断：

1. **判断翻译是否存在双关语**：
   - 对于同音双关，翻译中必须有一个词，存在一个同音或相似发音的词，且这两个词有不同的意义。若发音差距过大，请直接回答"No"。
   - 对于同形双关，翻译中必须有一个词，本身能理解出两个不同的意义，或者存在一个与之形状相同或相似的词，且这两个词有不同的意义。若没有理解出两个不同的意义，请直接回答"No"。

2. **分析双关语的合理性**：
   - 如果翻译中存在双关，分析其是否合适并且在中文中能够自然理解。
   - 对于同音双关，解释发音相似或相同的词，以及它们的不同含义。
   - 对于同形双关，解释词形相似的词，以及它们的不同含义。

3. **文化和语境的考量**：
   - 请确保判断中考虑中文母语者的理解和接受度。如果双关语在中文中不自然或无法产生有效的幽默或双重意义，请回答"No"。
   - 我们允许将原语言的同音双关翻译成一个同形双关，或者将原语言的同形双关翻译成一个同音双关。
   - 我们不允许通过括号注释来传达原语言双关语的含义，不允许直接将原语言的两重意思直接翻译出来或翻译成一个词，也不允许将原语言的双关语翻译成一个没有双关的句子。

Final Answer: Yes/No
'''

PROMPT = PROMPT_en


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Print the raw line for debugging
                #print(f"Line {line_num}: {repr(line)}")
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error at position {e.pos}: {line[max(0, e.pos-10):min(len(line), e.pos+10)]}")
                raise
    return data

def load_json(file_path: str) -> Dict:
    """Load data from a json file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(file_path):
    """Ensure output directory exists"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def check_pun_with_llm(translation: str, original_sentence: str, index: int) -> Dict:
    user_prompt = f"""
    The original sentence is:
    {original_sentence}
    Please carefully read the model output:
{translation}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": dedent(PROMPT)},
                {"role": "user", "content": user_prompt}
            ],
        )
        import re
        match = re.search(r'Final Answer:\s*(Yes|No)', response.choices[0].message.content, re.IGNORECASE)
        result = match.group(1).strip().lower() if match else 'no'
        return {
            'index': index,
            'hit': result == 'yes',
            'sentence': original_sentence,
            'translation': translation,
            'response': response.choices[0].message.content
        }
    except Exception as e:
        print(f"Error in LLM API call for index {index}: {e}")
        return {
            'index': index,
            'hit': False,
            'sentence': original_sentence,
            'translation': translation,
            'response': str(e)
        }

def process_batch(batch_items, original_data, executor):
    """Process a batch of data"""
    futures = []
    results = []
    
    try:
        for item in batch_items:
            index = item['index']
            # Assuming original_data is a list, access by index directly
            original_sentence = original_data[index] if index < len(original_data) else ""
            futures.append(
                executor.submit(
                    check_pun_with_llm,
                    item['translation'],
                    original_sentence,
                    index
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
    with open(file_path, 'w', encoding='utf-8') as file:  # Changed from 'a' to 'w' to overwrite
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(f"{json_line}\n")

def evaluate_translations(pred_file: str, original_file: str, output_file: str, batch_size: int = 10, max_workers: int = 5):
    """Evaluate translations for puns."""
    
    # Load predictions and original data
    pred_data = load_jsonl(pred_file)
    original_data = load_json(original_file)
    
    # 创建原始句子的索引映射
    original_sentences = {item['index']: item['sentence'] for item in original_data}
    
    # Prepare data for batch processing
    evaluation_data = []
    for item in pred_data:
        index = item.get('index', 0)
        translation = item.get('pred_tag', '')
        
        if not translation:
            print(f"Empty translation found for index {index} - skipping")
            continue
            
        if index not in original_sentences:
            print(f"No matching original sentence for index {index} - skipping")
            continue
            
        evaluation_data.append({
            'translation': translation,
            'index': index,
            'sentence': original_sentences[index]
        })
    
    print(f"\nEvaluation Setup:")
    print(f"Total translations for evaluation: {len(evaluation_data)}")
    
    # Split data into batches
    batches = [evaluation_data[i:i + batch_size] for i in range(0, len(evaluation_data), batch_size)]
    
    all_results = []
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_num, batch in enumerate(tqdm(batches), 1):
                try:
                    futures = []
                    for item in batch:
                        futures.append(
                            executor.submit(
                                check_pun_with_llm,
                                item['translation'],
                                item['sentence'],
                                item['index']
                            )
                        )
                    
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            all_results.append(result)
                        except Exception as e:
                            print(f"Error processing future: {str(e)}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing batch {batch_num}: {str(e)}")
                    continue
    
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Partial results have been saved.")
    finally:
        if all_results:
            # 按索引排序结果
            all_results.sort(key=lambda x: x['index'])
            write_jsonl_file(output_file, all_results)
    
    # Calculate hit rate
    total = len(all_results)
    hits = sum(1 for r in all_results if r['hit'])
    hit_rate = hits / total if total > 0 else 0
    
    return {
        'total': total,
        'hits': hits,
        'hit_rate': hit_rate
    }

def main():

    lan = 'zh'
    #lan = 'en'
    #type = 'phonic'
    type = 'graphic'

    #strategy = 'vanilla'
    #strategy = 'cot'
    strategy = 'cvo'

    #model = 'qwen-vl-max'
    model = '4o'
    #model = 'qwq'
    #model = 'o1-mini'

    
    # File paths
    pred_file = f"output/{model}/translation/new/{lan}_{type}_translation_{strategy}_post.jsonl"
    original_file = f"data/textual/{lan}/{type}.json"
    output_file = f"eval/output/{model}/translation/new/{lan}_{type}_translation_{strategy}_hit.jsonl"
    
    # Ensure output directory exists
    ensure_dir(output_file)
    
    # Run evaluation
    results = evaluate_translations(
        pred_file=pred_file,
        original_file=original_file,
        output_file=output_file,
        batch_size=20,
        max_workers=20
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total evaluated: {results['total']}")
    print(f"Total hits: {results['hits']}")
    print(f"Hit rate: {results['hit_rate']*100:.2f}%")
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
