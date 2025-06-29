import json
from tqdm import tqdm
from textwrap import dedent
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

'''client = OpenAI(api_key=" ", base_url=" ")
MODEL = "deepseek-v3"'''

client = OpenAI(api_key="sk-5wGTFItFpEsA0OoVyPCEC1ItJnyS0dsFtpUTvrWlzqimqgRb", base_url="https://www.DMXapi.com/v1/")
MODEL = "claude-3-5-sonnet-20241022"

definition = '''同音双关（homophonic pun）指两个词发音相同或相似，但意义不同，由此构成双关；
同形双关（homographic pun）指同一个词能够被理解为两重意思，或两个词形相同或相似，但意义不同，由此构成双关。
'''

PROMPT = f'''
{definition}
你的任务：
若原文是中文，请将这句双关语翻译成英文，并尽可能保持原文的双关效果或在目标语中再造一个双关。
若原文是英文，请将这句双关语翻译成中文，并尽可能保持原文的双关效果或在目标语中再造一个双关。
Let's think step by step like this (Use English indicators "Analysis:" and "Final Answer:", keep your final answer in one line, no further explanation after):

Analysis:
1. ...
2. ...
3. ...
...

Final Answer:
...
'''

def ensure_dir(file_path):
    """确保输出目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def gpt(input_sentence, index):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": dedent(PROMPT)
                },
                {
                    "role": "user",
                    "content": input_sentence
                }
            ],
        )
        answer = response.choices[0].message.content
        # Extract final answer using regex - get everything after the last "Final Answer:"
        import re
        final_answer = re.search(r'Final Answer:\s*(.+)$', answer, re.DOTALL) 
        if final_answer:
            pred_tag = final_answer.group(1).strip()
        else:
            # Fallback to full answer if no match found
            pred_tag = answer
            
        return {"index": index, "pred_tag": pred_tag}        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing index {index}: {error_msg}")
        # Return a dictionary with error information
        return {
            "index": index,
            "pred_tag": None, 
            "error": error_msg
        }

def load_existing_results(filename):
    """Load and parse existing results from the output file"""
    processed_indices = set()
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    result = json.loads(line.strip())
                    processed_indices.add(result['index'])
                except json.JSONDecodeError:
                    continue
    return processed_indices

def process_batch(batch_items, executor):
    """处理一批数据"""
    futures = []
    results = []
    
    try:
        # 提交所有任务
        for item in batch_items:
            sentence = item['sentence']
            index = item['index']
            futures.append(executor.submit(gpt, sentence, index))
        
        # 收集结果
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

def read_json_file(file_path):
    """读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_jsonl_file(file_path, data):
    """将数据以jsonl格式追加写入文件，每行一个条目"""
    ensure_dir(file_path)  # 确保目录存在
    with open(file_path, 'a', encoding='utf-8') as file:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(f"{json_line}\n")

def main(input_json_path, output_jsonl_path, batch_size=10, max_workers=5):
    # 确保输出目录存在
    ensure_dir(output_jsonl_path)
    
    # 读取已处理的索引
    processed_indices = load_existing_results(output_jsonl_path)
    print(f"Found {len(processed_indices)} processed indices")
    
    # 读取JSON文件
    input_data = read_json_file(input_json_path)
    
    # 过滤已处理的数据
    filtered_data = [item for item in input_data if item['index'] not in processed_indices]
    print(f"Processing {len(filtered_data)} remaining items")
    
    if not filtered_data:
        print("All items have been processed. Nothing to do.")
        return
    
    # 将数据分成批次
    batches = [filtered_data[i:i + batch_size] for i in range(0, len(filtered_data), batch_size)]
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_num, batch in enumerate(tqdm(batches), 1):
                try:
                    # 处理当前批次
                    results = process_batch(batch, executor)
                    # Sort results by index
                    results.sort(key=lambda x: x['index'])
                    
                    # 写入结果
                    if results:
                        write_jsonl_file(output_jsonl_path, results)
                        print(f"Batch {batch_num} saved with {len(results)} records.")
                    
                except KeyboardInterrupt:
                    print("\nDetected keyboard interrupt, saving current results...")
                    raise
                except Exception as e:
                    print(f"Error processing batch {batch_num}: {str(e)}")
                    continue
    
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Partial results have been saved.")
        return
    
    print("Processing completed successfully!")

# 设置输入输出文件路径
lan = 'zh'
#lan = 'en'
#type = 'phonic'
type = 'graphic'

model = 'claude'
#model = '4o'
#model = 'qwq'
#model = 'o1-mini'

input_json_path = f'data/textual/{lan}/{type}.json'
output_jsonl_path = f'output/{model}/translation/{lan}_{type}_translation_cot.jsonl'

if __name__ == "__main__":
    try:
        main(input_json_path, output_jsonl_path, batch_size=10, max_workers=10)
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
