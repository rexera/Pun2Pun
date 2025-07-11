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

CVO = '''
这是一个从源语言双关翻译到目标语言双关的翻译任务。请主要关注翻译中的“词语选用”。
在这个任务中，你需要理解并应用“常量-变量”理论来提升双关语翻译的效果。以下是任务的具体步骤和三个常量、三个变量的定义，以便你能准确完成任务。

### 常量-变量理论简介

**常量**和**变量**是翻译中用于分析双关语结构的基本元素。双关语在源语言和目标语言中常常通过不同的词语组合来实现，为了准确保留其含义，模型需要对常量和变量进行分解和匹配。

#### 三个常量（Source Meanings, SMs）

1. **常量1 (SM1)**：这是源语言中**双关所在的核心词或词组**，是承载双关效果的词。它包含了词义/语义上的双重含义。
  - 这是1个词/词组。写作：[SM1]

2. **常量2 (SM2)**：由两个要素构成：
   - **A**：常量2的依据（Anchor），即引导读者识别出双关含义的语义基础，通常是直接联想到双关含义的关键概念或语义联想。
   - **B**：支撑词（Bridge），与常量1共同构成双关语义。

   **写作形式**：常量2表示为[A, B]。

3. **常量3 (Source Pragmatic Meaning, SPM)**：这是**源语言中整体双关效果的语用含义**，由常量1和常量2支撑词（Bridge）的组合所构成。
  - 这是一对词。写作: [SM1 + B]

#### 三个变量（Target Meanings, TMs）

1. **变量1 (TM1)**：目标语言中**围绕源语言常量1**枚举出的一个核心词或词组。它应当能再现源语言的双重含义，是目标语言双关结构的基础。
  - 这是一个词/词组。写作：[TM1]

2. **变量2 (TM2)**：在目标语言中为双关提供支持，与源语言中的常量2相对应。它通常有两个可能性：
   - 综合了常量2 (SM2) 的两个含义。
   - 在某些情况下仅选择其中一个含义，以确保双关效果的自然表达。
   - 这是一个词/词组。写作：[TM2]

3. **变量3 (TPM)**：目标语言中再现双关整体效果的语用含义。它考虑了变量1和变量2的含义，在目标语言中再现源语言的双重含义（TPM1、TPM2）和双关修辞效果。
   - 这是一对词，写作：[TPM1,TPM2]
   - 如果达到了“谐音”效果，应为谐音的两个词。如：[嗅, 锈]
   - 如果达到了“谐义”效果，应为该词的两个意义。如：[“金钱”豹, “钱”的味道]
   
   你需要做的是：1. 识别三个常量；2.围绕常量枚举变量；3. 在变量中调整用词，朝着SM1-TM1、SM2-TM2、SPM-TPM三组量的完美匹配靠拢。
   
'''

PROMPT = f'''
{definition}

下面的常量变量优化理论可以帮助你完成任务：
{CVO}

你的任务：

请根据理论，判断这一句是同音双关还是同形双关。同音双关输出phonic，同形双关输出graphic。仅需要输出一个单词。
Let's think step by step like this (Use English indicators "Analysis:" and "Final Answer:"):

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
        final_answer = re.search(r'Final Answer:\s*(.+?)(?=\n|$)', answer, re.DOTALL)
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
#lan = 'zh'
lan = 'en'
#type = 'phonic'
type = 'graphic'

#model = 'qwen-vl-max'
model = 'claude'
#model = 'qwq'
#model = 'o1-mini'

input_json_path = f'data/textual/{lan}/{type}.json'
output_jsonl_path = f'output/{model}/tag/{lan}_{type}_tag_cvo.jsonl'

if __name__ == "__main__":
    try:
        main(input_json_path, output_jsonl_path, batch_size=10, max_workers=10)
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
