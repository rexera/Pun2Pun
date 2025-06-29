import json
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize client
client = OpenAI(api_key="", base_url="")

# Prompts
ovl_theory = """
你是一个非常严苛的评分专家，负责对中英双关语翻译的质量进行评分。请不要过于礼貌或手下留情。
这是一个从源语言双关翻译到目标语言双关的翻译任务。请主要关注翻译中的"词语选用"，不要过分分析内容和主题。

我们对于"双关"的定义如下：
同音双关（homophonic pun）指两个词发音相同或相似，但意义不同，由此构成双关；
同形双关（homographic pun）指同一个词能够被理解为两重意思，或两个词形相同或相似，但意义不同，由此构成双关。

在这个任务中，你需要理解并应用"常量-变量"理论来评估双关语翻译的效果。以下是任务的具体步骤和三个常量、三个变量的定义，以便你能准确完成任务。

注意：
输入给你的原始句子中【全部都有双关】，请认真分析，不要逃避。
但是输入给你的【模型翻译结果】中可能没有双关/不符合我们对双关的定义。

### 常量-变量理论简介

**常量**和**变量**是翻译中用于分析双关语结构的基本元素。双关语在源语言和目标语言中常常通过不同的词语组合来实现，为了准确保留其含义，模型需要对常量和变量进行分解和匹配。

#### 三个【从原句中得到的】常量（Source Meanings, SMs）

1. **常量1 (SM1)**：这是源语言中**双关所在的核心词或词组**，是承载双关效果的词。它包含了词义/语义上的双重含义。
  - 这是1个词/词组。写作：[SM1]

2. **常量2 (SM2)**：由两个要素构成：
   - **A**：常量2的依据（Anchor），即引导读者识别出双关含义的语义基础，通常是直接联想到双关含义的关键概念或语义联想。
   - **B**：支撑词（Bridge），与常量1共同构成双关语义。

   **写作形式**：常量2表示为[A, B]。

3. **常量3 (Source Pragmatic Meaning, SPM)**：这是**源语言中整体双关效果的语用含义**，由常量1和常量2支撑词（Bridge）的组合所构成。
  - 这是一对词。写作: [SM1 + B]

#### 三个【从译句中得到的】变量（Target Meanings, TMs）

1. **变量1 (TM1)**：目标语言中**围绕源语言常量1**[枚举]出的一个核心词或词组。它应当能再现源语言的双重含义，是目标语言双关结构的基础。
  - 这是一个词/词组。写作：[TM1]

2. **变量2 (TM2)**：在目标语言中为双关提供支持，与源语言中的常量2相对应。它通常有两个可能性：
   - 综合了常量2 (SM2) 的两个含义。
   - 在某些情况下仅选择其中一个含义，以确保双关效果的自然表达。
   - 这是一个词/词组。写作：[TM2]
   - TM2 应围绕着 SM2 枚举。

3. **变量3 (TPM)**：目标语言中再现双关整体效果的语用含义。它考虑了变量1和变量2的含义，在目标语言中再现源语言的双重含义（TPM1、TPM2）和双关修辞效果。
   - 这是一对词，写作：[TPM1,TPM2]
   - 如果达到同音双关，应为谐音的两个词。如：[嗅, 锈]
   - 如果达到同形双关，应为同形词的两个意义。如：["金钱"豹, "钱"的味道]
   - TPM 不应该是 SPM 的简单翻译，而应该是在目标语内再造的一个双关的两个部分。

### 重叠度打分

为了衡量源语言的常量和目标语言的变量之间的对应程度，我们使用重叠度评分。评分基于以下三对：<SM1-TM1>、<SM2-TM2>、<SPM-TPM>，并且评分的区间为0-100，分数越高表示目标语言对源语言的语义和双关效果保留得越完整。

---

### 下面是一个示例

**原文**：
- A: What animal is rich?
- B: Bloodhound, because he is always picking up scents.

1. **常量1：[scents]**
    - **来源**：在原文中，"scents"这个词具有双关性质，既指"气味"（表层含义）又暗指"钱"（通过与"cents"谐音实现的隐含含义）。因此，常量1就是这个承载双关意义的词"scents"。
    - **双关功能**：常量1的双重含义为整个双关效果提供了基础。
2. **常量2：[rich, cents]**
    - **来源**：常量2的作用是帮助读者识别到常量1的隐含含义。为了实现这一点，常量2分为两个部分：
        - **依据（A）**：即常量2的语义联想基础，可以让译者联想到"钱"的隐含含义。在这里，"rich"的语义使得联想到"金钱"。
        - **支持词（B）**：与常量1组合成双关效果的词。在这个例子中，"cents"是常量2的支持词（B），帮助"scents"产生"气味"和"金钱"的双关效果。
3. **常量3：[scents + cents]**
    - **来源**："scents + cents"的同音构成的双关幽默修辞效果。

**翻译1**：
- A: 什么动物很有钱？
- B: 金钱豹，它身上全是金钱。
  - TM1: []
  - TM2: [有钱]
  - TPM: ["金钱"豹 + 金钱]

  - **评估**：
      - **<SM1-TM1>**：未保留"气味"这个层次。打分为0（没有再现双重含义）。
      - **<SM2-TM2>**：此翻译中的"金钱"部分在某种程度上暗示了"rich"的隐含语境，但缺乏"气味"的具体层次。打分为50（隐含意义再现不完整）。
      - **<SPM-TPM>**：该翻译的语用效果单一，仅传达了"金钱"的概念，而未达到双关效果中"气味-金钱"双重含义的结合，因此语用效果偏低。打分为40。

**翻译2**：
- A: 什么动物很富有？
- B: 金钱豹，走几步都是钱的味道。
  - TM1: [味道]
  - TM2: [富有]
  - TPM: ["金钱"豹 + "钱"的味道]

  - **评估**：
      - **<SM1-TM1>**：此翻译通过"味道"保留了原句中"气味"的意义。打分为90。
      - **<SM2-TM2>**："富有"更体现一种"行事风格"，能和"味道"结合，较好地传达了次要含义的联想效果。打分为80。
      - **<SPM-TPM>**：该翻译在目标语言中实现了双关的语用效果，保留了双重含义，使得"味道"与"钱"之间的双关效果在语用层面符合中文表达习惯。打分为90。

这个例子展示了翻译2在保留双关效果和语用含义上的优势，并说明了打分的依据。
"""

format = '''{"SM1": "scents", "SM2": "rich, cents", "SPM": "scents + cents", "TM1": "气味", "TM2": "金钱", "TPM": "嗅, 锈"}'''

# Step 1: Extract 3 Pairs
step_1_system = f'''请首先阅读以下理论：
---------------------------------
{ovl_theory}
---------------------------------

你的任务：

请分析以下双关语的原文和翻译，识别出所有常量和变量。仅输出一个JSON对象，包含以下字段：

    "SM1": str,  
    "SM2": str,  
    "SPM": str,  
    "TM1": str,  
    "TM2": str,  
    "TPM": str   
---------------------------------
下面是两个示例：

原文：
- A: What animal is rich?
- B: Bloodhound, because he is always picking up scents.

翻译：
- A: 什么动物很富有？
- B: 金钱豹，走几步都是钱的味道。

"SM1": "scents", "SM2": "rich, cents", "SPM": "scents + cents", "TM1": "气味", "TM2": "金钱", "TPM": "嗅, 锈"

原文：
''3.14159265,'' Tom said piously.

翻译：
''3.14159265,'' 汤姆虔诚地说，仿佛在念老天"π"的经。

"SM1": "piously", "SM2": "3.14159265, pi", "SPM": "piously + pi", "TM1": "虔诚地", "TM2": "π经", "TPM": "π, 派"
---------------------------------

最后输出一行jsonl，不需要```json```包裹。
注意：我们允许homophonic pun和homographic pun在翻译时类型转换，请注意识别翻译句子中是否存在类型转换，不要误判为没有双关，请将上述字段全部输出，不要遗漏。
请Analyze step by step，输出格式如：（请使用英文的提示语"Analysis""Extraction"，提示语不要用**包裹，提取结果不需要```jsonl```包裹）

Preliminaries:

This is a [homophonic/homographic] pun, playing on the [homophonic/homographic] relationship between [SPM1] and [SPM2].

Now, for three source meanings:

Analysis:
1. SM1: ...
2. SM2: ...
3. SPM: ...
...

Now, for three target meanings:

Analysis:
1. TM1: ...(how it came into being through enumeration)
2. TM2: ...
3. TPM: ...(how the two parts constitute homophonic/homographic pun)
...

Extraction:
{format}
'''

step_1_user = '''
Source: {source}
Translation: {translation}
'''

format_1 = '{"ovl1": float, "ovl2": float, "ovl3": float}'

step_2_system = f'''请首先阅读以下理论：
---------------------------------
{ovl_theory}
---------------------------------

你的任务：

根据已经提取出来的三对的具体内容，分别评估<SM1-TM1>、<SM2-TM2>、<SPM-TPM>的重叠度。评分标准如下：

1. <SM1-TM1> 评分标准 (0-100)：
   - 90-100：完全保留了原文双关词的双重含义，且表达自然
   - 70-89：基本保留了双重含义，但表达略显生硬
   - 40-69：仅保留了部分含义
   - 0-39：完全丧失双关词的双重含义

2. <SM2-TM2> 评分标准 (0-100)：
   - 90-100：完全保留了原文的语境支撑和语义联想
   - 70-89：基本保留了语境支撑，但联想稍弱
   - 40-69：语境支撑不完整
   - 0-39：完全丧失语境支撑作用

3. <SPM-TPM> 评分标准 (0-100)：
   - 90-100：完美重现双关效果，且符合目标语言表达习惯
   - 70-89：成功构建双关，但略显生硬
   - 40-69：双关效果较弱，或表达不自然
   - 0-39：未能构建双关效果

再次提醒，请严格评分，压低整体分数。

请Analyze step by step，输出格式如：（请使用英文的提示语"Analysis""Scores"，提示语不要用**包裹，最终分数不需要```jsonl```包裹）

Analysis for SM1-TM1:
1. ...
2. ...
...
ovl1: ...   

Analysis for SM2-TM2:
1. ...
2. ...
...
ovl2: ...

Analysis for SPM-TPM:
1. ...
2. ...
...
ovl3: ...

Scores:
{format_1}
'''

step_2_user = '''
Source: {source}
Translation: {translation}
Extraction: {cvo}
'''


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


def get_cvo_analysis(question: Dict, translation: Dict) -> Dict:
    """Get CVO analysis from model"""
    prompt = step_1_user.format(
        source=question['sentence'],
        translation=translation['pred_tag']
    )
    
    response = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[
            {"role": "system", "content": step_1_system},
            {"role": "user", "content": prompt}
        ],
    )
    import re

    # Extract the JSON part from the response
    match = re.search(r'Extraction:\n({.*})', response.choices[0].message.content, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            cvo_analysis = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Extracted content is not a valid JSON object")
    else:
        raise ValueError("No valid JSON object found in the response")
    print(cvo_analysis)
    return cvo_analysis

def evaluate_translation(question: Dict, translation: Dict) -> Dict:
    """Evaluate a single translation"""
    try:
        # Get CVO analysis
        cvo = get_cvo_analysis(question, translation)
        
        # Get direct scoring from the second model
        prompt = step_2_user.format(
            source=question['sentence'],
            translation=translation['pred_tag'],
            cvo=cvo
        )
        
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[
                {"role": "system", "content": step_2_system},
                {"role": "user", "content": prompt}
            ],
        )
        
        # Extract scores from response using more robust regex
        import re
        scores_match = re.search(r'Scores:\s*({.*?})\s*$', response.choices[0].message.content, re.DOTALL)
        if not scores_match:
            raise ValueError(f"No scores found in response: {response.choices[0].message.content}")
            
        scores_str = scores_match.group(1)
        # Clean up the JSON string to ensure proper formatting
        scores_str = scores_str.replace("'", '"').strip()
        try:
            scores = json.loads(scores_str)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {scores_str}")
            raise ValueError(f"Failed to parse scores JSON: {str(e)}")
        
        # Calculate final score with weights
        final_score = (scores['ovl1'] * 0.25 + 
                      scores['ovl2'] * 0.25 + 
                      scores['ovl3'] * 0.5)
        
        return {
            "index": question["index"],
            "sentence": question["sentence"],
            "translation": translation["pred_tag"],
            "SM1": cvo["SM1"],
            "SM2": cvo["SM2"], 
            "SPM": cvo["SPM"],
            "TM1": cvo["TM1"],
            "TM2": cvo["TM2"],
            "TPM": cvo["TPM"],
            "ovl1": scores["ovl1"],
            "ovl2": scores["ovl2"],
            "ovl3": scores["ovl3"],
            "score": final_score,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error in evaluate_translation: {str(e)}")  # Add debug output
        return {
            "index": question["index"],
            "error": str(e),
            "status": "error"
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

def load_data(question_path: str, hit_path: str, translation_path: str) -> Tuple[List[Dict], Dict, List[Dict]]:
    """Load all required data files"""
    with open(question_path, 'r') as f:
        questions = json.load(f)
    
    with open(hit_path, 'r') as f:
        hits = {item['index']: item['hit'] for item in [json.loads(line) for line in f]}
    
    with open(translation_path, 'r') as f:
        translations = [json.loads(line) for line in f]
    
    return questions, hits, translations

def ensure_dir(file_path):
    """确保目录存在"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def write_json_file(file_path: str, data):
    """将数据以json格式写入文件"""
    ensure_dir(file_path)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print("Data written successfully")  # Debug output

def load_existing_results(output_path: str) -> List[Dict]:
    """加载已有的结果"""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            print(f"Warning: Could not load existing results from {output_path}")
    return []

def main():
    batch_size = 10
    max_workers = 10

    #lan = 'zh'
    lan = 'en'
    #type = 'phonic'
    type = 'graphic'

    #strategy = 'vanilla'
    #strategy = 'cot'
    strategy = 'cvo'

    #model = 'qwen-vl-max'
    #model = '4o'
    #model = 'qwq'
    model = 'o1-mini'

    # File paths
    question_path = f"data/textual/{lan}/{type}.json"
    hit_path = f"eval/output/{model}/translation/new/{lan}_{type}_translation_{strategy}_hit.jsonl"
    translation_path = f"output/{model}/translation/new/{lan}_{type}_translation_{strategy}_post.jsonl"
    output_path = f"eval/output/{model}/translation/new/{lan}_{type}_translation_{strategy}_ovl_sub.json"
    
    # 确保输出目录存在
    ensure_dir(output_path)
    
    # Load data
    questions, hits, translations = load_data(question_path, hit_path, translation_path)
    
    # Load existing results
    results = load_existing_results(output_path)
    evaluated_indices = {r["index"] for r in results}
    print(f"Found {len(evaluated_indices)} processed indices")
    
    # Create pairs of items to process
    items_to_process = []
    for question, translation in zip(questions, translations):
        if question["index"] not in evaluated_indices and hits.get(question["index"], False):
            items_to_process.append((question, translation))
    
    print(f"Processing {len(items_to_process)} remaining items")
    
    if not items_to_process:
        print("All items have been processed. Computing average from existing results...")
    else:
        # Process in batches
        batches = [items_to_process[i:i + batch_size] for i in range(0, len(items_to_process), batch_size)]
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for batch_num, batch in enumerate(tqdm(batches), 1):
                    try:
                        batch_results = process_batch(batch, executor)

                        if batch_results:
                            # Sort and extend results
                            batch_results.sort(key=lambda x: x['index'])
                            results.extend(batch_results)
                            
                            # Save after each batch
                            try:
                                write_json_file(output_path, results)
                                print(f"Batch {batch_num} saved with {len(batch_results)} new records. Total: {len(results)}")
                            except Exception as e:
                                print(f"Error saving results: {str(e)}")
                    
                    except KeyboardInterrupt:
                        print("\nDetected keyboard interrupt, saving current results...")
                        raise
                    except Exception as e:
                        print(f"Error processing batch {batch_num}: {str(e)}")
                        continue
                    
        except KeyboardInterrupt:
            print("\nScript interrupted by user. Partial results have been saved.")
            return
        
    # Calculate final statistics regardless of whether new processing occurred
    hit_indices = {idx for idx, h in hits.items() if h}
    # Only include successful evaluations in score calculation
    hit_scores = [r["score"] for r in results 
                 if r["index"] in hit_indices and 
                 (r.get("status", "success") == "success") and  # Default to "success" for backward compatibility
                 "score" in r]
    
    if hit_scores:
        final_avg_score = sum(hit_scores) / len(hit_scores)
        print(f"\nStatistics:")
        print(f"Final average score: {final_avg_score:.2f}")
        print(f"Successful evaluations: {len(hit_scores)}")
        print(f"Total results: {len(results)}")
        print(f"Failed evaluations: {len(results) - len(hit_scores)}")
    else:
        print("\nNo valid scores found to calculate average.")

if __name__ == "__main__":
    main()
