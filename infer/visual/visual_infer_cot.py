import base64
import json
from tqdm import tqdm
from textwrap import dedent
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

client = OpenAI(api_key=" ", base_url=" ")
MODEL = "qvq-72b-preview"

definition = '''同音双关（homophonic pun）指两个词发音相同或相似，但意义不同，由此构成双关；
同形双关（homographic pun）指同一个词能够被理解为两重意思，或两个词形相同或相似，但意义不同，由此构成双关。
'''

TAG_PROMPT = f'''
{definition}
请判断图中的这一句是同音双关还是同形双关。同音双关输出phonic，同形双关输出graphic。仅需要输出一个单词。最终答案后不需要解释。
Let's think step by step:

Analysis:
1. ...
2. ...

Final Answer:
...
'''

EXPLANATION_PROMPT = f'''
{definition}
请解释这句双关语的机制，同音双关需说明如何读音相近或相同，同形双关需说明如何构成一词多义。最终答案后不需要解释。
Let's think step by step:

Analysis:
1. ...
2. ...

Final Answer:
...
'''

APPRECIATION_PROMPT = f'''
{definition}
请用一句话解释这句双关语的图文关联、文化背景和使用场景。最终答案（一句话）后不需要解释。
Let's think step by step:

Analysis:
1. ...
2. ...

Final Answer:
...
'''

TRANSLATION_PROMPT = f'''
{definition}
你的任务：根据图文语境，
请将这句双关语翻译成中文，并尽可能保持原文的双关效果或在目标语中再造一个双关。最终答案后不需要解释。
Let's think step by step:

Analysis:
1. ...
2. ...

Final Answer:
...
'''

'''若原文是中文，请将这句双关语翻译成英文，并尽可能保持原文的双关效果或在目标语中再造一个双关。最终答案后不需要解释。
若原文是英文，请将这句双关语翻译成中文，并尽可能保持原文的双关效果或在目标语中再造一个双关。最终答案后不需要解释。'''

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_image_files(directory):
    """获取指定目录下的所有jpg图片并按文件名数字排序"""
    image_files = glob(os.path.join(directory, "*.jpg"))
    return sorted(image_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

def get_image_index(image_path):
    """从图片文件名中提取数字作为索引"""
    return int(os.path.basename(image_path).split('.')[0])

def extract_final_answer(response):
    """Extract final answer using regex - get everything after the last "Final Answer:" """
    import re
    final_answer = re.search(r'Final Answer:\s*([\s\S]+)$', response)
    if final_answer:
        return final_answer.group(1).strip()
    return response

def gpt(image_path, prompt):
    """处理单个图片文件的单个任务"""
    try:
        base64_image = encode_image(image_path)
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        }
                    ],
                }
            ],
        )
        response = completion.choices[0].message.content
        return extract_final_answer(response)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing image {image_path}: {error_msg}")
        return None

def process_single_image(image_path):
    """处理单个图片的所有任务"""
    try:
        index = get_image_index(image_path)
        
        # 依次执行四个任务
        tag = gpt(image_path, TAG_PROMPT)
        explanation = gpt(image_path, EXPLANATION_PROMPT)
        appreciation = gpt(image_path, APPRECIATION_PROMPT)
        translation = gpt(image_path, TRANSLATION_PROMPT)
        
        return {
            "index": index,
            "tag": tag,
            "explanation": explanation,
            "appreciation": appreciation,
            "translation": translation
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error processing image {image_path}: {error_msg}")
        return {
            "index": index,
            "tag": None,
            "explanation": None,
            "appreciation": None,
            "translation": None,
            "error": error_msg
        }

def ensure_dir(file_path):
    """确保输出目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

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
    """处理一批图片数据"""
    futures = []
    results = []
    
    try:
        # 提交所有任务
        for image_path in batch_items:
            futures.append(executor.submit(process_single_image, image_path))
        
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

def main(input_image_dir, output_jsonl_path, batch_size=10, max_workers=5):
    # 确保输出目录存在
    ensure_dir(output_jsonl_path)
    
    # 读取已处理的索引
    processed_indices = load_existing_results(output_jsonl_path)
    print(f"Found {len(processed_indices)} processed indices")
    
    # 获取所有图片文件
    image_files = get_image_files(input_image_dir)
    print(f"Found {len(image_files)} image files")
    
    # 过滤已处理的图片
    items_to_process = [
        path for path in image_files
        if get_image_index(path) not in processed_indices
    ]
    
    if not items_to_process:
        print("All items have been processed. Nothing to do.")
        return
    
    print(f"Processing {len(items_to_process)} remaining items")
    
    # 将数据分成批次
    batches = [items_to_process[i:i + batch_size] for i in range(0, len(items_to_process), batch_size)]
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_num, batch in enumerate(tqdm(batches), 1):
                try:
                    results = process_batch(batch, executor)
                    results.sort(key=lambda x: x['index'])
                    
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
#lan = 'zh'
lan = 'en'

#model = 'qwen-vl-max'
#model = '4o'
model = 'qvq'
#model = 'o1-mini'

# 设置输入输出路径
input_image_dir = f"data/visual/pun2pun_image/{lan}"
output_jsonl_path = f"visual_output/{model}/{lan}_cot.jsonl"

if __name__ == "__main__":
    try:
        main(input_image_dir, output_jsonl_path, batch_size=10, max_workers=10)
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
