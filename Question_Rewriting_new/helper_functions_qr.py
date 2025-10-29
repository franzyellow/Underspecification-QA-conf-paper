import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
from typing import List, Literal, List, Dict, Any, Optional
import numpy as np
import seaborn as sns

from datasets import load_dataset
import random
import json
import re
from functools import partial
from datasets import Dataset
from copy import deepcopy
import evaluate
import nltk
from scipy.stats import ttest_ind
import string
from collections import Counter

import openai
import os
import time
import pandas as pd
import torch

from ragas.llms import LangchainLLMWrapper
from langchain_deepseek import ChatDeepSeek
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import AnswerAccuracy
from dotenv import load_dotenv
load_dotenv()

def modify_question(question, short_answer, reasoning, client, model, temperature=0, max_retries=5, sleep_time=2.0):
    system_prompt = (
        "You are a professional question optimization expert. Please modify the underspecified question to a fully specified version based on the provided clues.\n\n"
        "Requirements:\n"
        "1. Keep the core intent of the question unchanged\n"
        "2. Add necessary contextual information\n"
        "3. Eliminate underspecified elements and make the question clear\n"
        "4. Ensure the modified question can be directly answered with the provided short answer without dispute\n\n"
        "Please only return the modified question, do not include any other explanations."
    )

    user_prompt = f"""
    The original question: {question}
    Short answer: {short_answer}
    Reasoning: {reasoning}

    Please analyze the underspecified elements in the original question, then modify the question to a fully specified version based on both the short answer and reasoning.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            content = response.choices[0].message.content
            modified_question = content.strip()
            return modified_question
        except Exception as e:
            retries += 1
            print(f"Attempt {retries} failed: {str(e)}")
            if retries < max_retries:
                print(f"Waiting {sleep_time * retries} seconds before retry...")
                time.sleep(sleep_time * retries)
            else:
                print(f"All retries failed, returning original question")
                return question  # If error occurs, return original question

def modification_in_batch(input_file, output_file, ref_col, client, model, batch_size=3):
    """
    按批次处理所有样本，提高处理效率
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        batch_size: 每批处理的样本数量
    
    Returns:
        list: 所有处理过的样本
    """
    
    all_processed_samples = []
    
    # Loading all the data from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    print(f"Total samples to process: {total_samples}")
    print(f"Batch size: {batch_size}")
    
    # Process all samples in batches
    for batch_start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_lines = lines[batch_start:batch_end]
        
        batch_processed_samples = []
        
        # Process each sample in the current batch
        for i, line in enumerate(batch_lines):
            try:
                data = json.loads(line.strip())
                
                # Extract necessary fields
                question = data['question']
                short_answer = data[ref_col]
                classifier_response = data['qwen3_model_response']
                classifier_reasoning = json.loads(classifier_response)['reasoning']
                
                # Modify questions
                modified_question = modify_question(question, short_answer, classifier_reasoning, client, model)
                
                # Create new data structure
                new_sample = {
                    'original_question': question,
                    'modified_question': modified_question,
                    'short_answer': short_answer,
                    'model_original_answer': data.get('model_short_answer', 'undefined'),
                    'classifier_reasoning': classifier_reasoning,
                    'original_f1': data.get('f1', 'undefined'),
                    'original_em': data.get('em', 'undefined'),
                    'original_AA': data.get('ragas_AA_short', 'undefined')
                }
                
                batch_processed_samples.append(new_sample)
                
                # Add delay to avoid API rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing sample {batch_start + i + 1}: {e}")
                # Create error sample to maintain consistency
                error_sample = {
                    'original_question': question,
                    'modified_question': modified_question,
                    'short_answer': short_answer,
                    'model_original_answer': data.get('model_short_answer', 'error'),
                    'classifier_reasoning': classifier_reasoning,
                    'original_f1': data.get('f1', 'error'),
                    'original_em': data.get('em', 'error'),
                    'original_AA': data.get('ragas_AA_short', 'error')
                }
                batch_processed_samples.append(error_sample)
        
        # Add batch results to all processed samples
        all_processed_samples.extend(batch_processed_samples)
        
        # Write intermediate results to file (append mode)
        with open(output_file, 'a', encoding='utf-8') as f:
            for sample in batch_processed_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
    
    print(f"\nAll batch processing completed! Total processed: {len(all_processed_samples)} samples")
    print(f"Results saved to: {output_file}")
    
    return all_processed_samples

def find_failed_rows_simple(input_file, output_file):
    """
    简单方法：通过比较原问题和修改后问题是否相同来找出失败的行
    """
    print("=== 查找失败的行（简单方法）===")
    
    # 读取输入和输出文件
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = [json.loads(line.strip()) for line in f]
    
    with open(output_file, 'r', encoding='utf-8') as f:
        output_data = [json.loads(line.strip()) for line in f]
    
    failed_rows = []
    
    for i, (input_row, output_row) in enumerate(zip(input_data, output_data)):
        original_question = input_row.get('question', '')
        modified_question = output_row.get('modified_question', '')
        
        # 如果原问题和修改后问题相同，说明失败了
        if original_question == modified_question:
            failed_rows.append({
                'row_number': i + 1,
                'original_question': original_question,
                'short_answer': input_row.get('short_answers', ''),
                'reasoning': input_row.get('reasoning', '')
            })
    
    print(f"发现 {len(failed_rows)} 个失败的行:")
    for row in failed_rows:
        print(f"\n第 {row['row_number']} 行:")
        print(f"  问题: {row['original_question'][:100]}...")
        print(f"  短答案: {row['short_answer']}")
        print(f"  推理: {row['reasoning'][:100]}...")
    
    return failed_rows

def ask_short_answer(question, client, model="gpt-4o-2024-11-20", temperature=0, max_retries=5, sleep_time=2.0):
    system_prompt = (
        "Answer the question with a concise response. "
        "Return answers as a list of strings. If there's only one answer, return a single-item list. "
        "Each answer should be brief and direct."
    )
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=temperature
            )
            content = response.choices[0].message.content
            if content.startswith("["):
                return eval(content)
            else:
                return [content.strip()]
        except Exception as e:
            retries += 1
            time.sleep(sleep_time * retries)
            
    return ["[Error]: Max retries exceeded"]


def run_batch_shortQA_api(batch, client, **kwargs):
    short_answers = []
    for q in batch["modified_question"]:
        try:
            answer = ask_short_answer(q, client=client, **kwargs)
            short_answers.append(answer)
        except Exception as e:
            print(f"Error: {e}")
            short_answers.append(["error"])
    return {"model_new_answer": short_answers}


def batch_QA_with_progress(dataset, batch_fn, output_key, batch_size=10, fill_value="error", **batch_fn_kwargs):
    all_outputs = []
    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Running {output_key}"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        try:
            output = batch_fn(batch, **batch_fn_kwargs)
            if output_key not in output:
                raise ValueError(f"Missing key '{output_key}' in batch result")
            all_outputs.extend(output[output_key])
        except Exception as e:
            print(f"Batch error at {i}: {e}")
            all_outputs.extend([fill_value] * len(batch))

    if len(all_outputs) != len(dataset):
        print(f"[Warning] Output length mismatch, auto-filling")
        all_outputs.extend([fill_value] * (len(dataset) - len(all_outputs)))

    return {output_key: all_outputs}

def evaluate_squad_per_sample_multi_ref_pred(dataset, pred_col="model_new_answer", ref_col="short_answer"):
    """
    对每个样本逐一计算 EM 和 F1，支持多个参考答案和多个预测答案（list[str]）。
    返回带 "em", "f1" 列的新 Dataset，以及 f1/em 列表用于统计分析。
    Also considering multiple answers in both gold and pred and take the maximum score
    """

    def normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            return ''.join(ch for ch in text if ch not in string.punctuation)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_exact(a_pred, a_gold):
    # 如果是 list，转成 set 并 normalize 每个元素
        if isinstance(a_pred, list) and isinstance(a_gold, list):
          pred_set = set(normalize_answer(a) for a in a_pred)
          gold_set = set(normalize_answer(a) for a in a_gold)
          return int(pred_set == gold_set)
        else:
          return int(normalize_answer(a_pred) == normalize_answer(a_gold))

    def compute_f1(a_pred, a_gold):
        pred_tokens = normalize_answer(a_pred).split()
        gold_tokens = normalize_answer(a_gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    new_data = []
    f1_scores = []
    em_scores = []

    for item in dataset:
        preds = item.get(pred_col, [])
        golds = item.get(ref_col, [])
        # 转为 list
        if not isinstance(preds, list):
            preds = [preds] if preds else []
        if not isinstance(golds, list):
            golds = [golds] if golds else []

        # 多对多最大匹配
        if not preds or not golds:
            em = 0.0
            f1 = 0.0
        else:
            em = max(compute_exact(p, g) for p in preds for g in golds)
            f1 = max(compute_f1(p, g) for p in preds for g in golds)

        new_item = deepcopy(item)
        new_item["new_em"] = em
        new_item["new_f1"] = f1
        new_data.append(new_item)
        em_scores.append(em)
        f1_scores.append(f1)

    return Dataset.from_list(new_data), f1_scores, em_scores


async def answer_accuracy(input_dataset, evaluator_llm, long_answer=False, ref_col = "short_answers"):
    # 在函数开始时创建一次 scorer
    scorer = AnswerAccuracy(llm=evaluator_llm)
    
    if long_answer:
        score_list_long = []
        score_list_short = []
        
        for i, row in enumerate(tqdm(input_dataset, desc="Calculating short and long answer accuracy")):
            try:
                # 长答案评分
                if 'model_long_answer' in row and 'long_answer' in row:
                    sample_long = SingleTurnSample(
                        user_input=row['question'],
                        response=row['model_long_answer'],
                        reference=row['long_answer']
                    )
                    score_long = await scorer.single_turn_ascore(sample_long)
                    score_list_long.append(score_long)
                else:
                    score_list_long.append(0.0)

                # 短答案评分 - 处理列表情况
                if 'model_short_answer' in row and ref_col in row:
                    model_answers = row['model_short_answer'] if isinstance(row['model_short_answer'], list) else [row['model_short_answer']]
                    reference_answers = row[ref_col] if isinstance(row[ref_col], list) else [row[ref_col]]
                    
                    # 计算所有组合的分数，取最高分
                    max_score = 0.0
                    for model_ans in model_answers:
                        for ref_ans in reference_answers:
                            sample_short = SingleTurnSample(
                                user_input=row['question'],
                                response=model_ans,
                                reference=ref_ans
                            )
                            score = await scorer.single_turn_ascore(sample_short)
                            max_score = max(max_score, score)
                    
                    score_list_short.append(max_score)
                else:
                    score_list_short.append(0.0)
                
            except Exception as e:
                print(f"处理第 {i+1} 个样本时出错: {e}")
                score_list_long.append(0.0)
                score_list_short.append(0.0)

        ragas_scored_dataset = input_dataset.add_column("ragas_AA_long", score_list_long)
        ragas_scored_dataset = ragas_scored_dataset.add_column("ragas_AA_short", score_list_short)

        return ragas_scored_dataset
    else:
        score_list = []
        
        for i, row in enumerate(tqdm(input_dataset, desc="Calculating short answer accuracy")):
            try:
                # 短答案评分 - 处理列表情况
                if 'model_short_answer' in row and ref_col in row:
                    model_answers = row['model_short_answer'] if isinstance(row['model_short_answer'], list) else [row['model_short_answer']]
                    reference_answers = row[ref_col] if isinstance(row[ref_col], list) else [row[ref_col]]
                    
                    # 计算所有组合的分数，取最高分
                    max_score = 0.0
                    for model_ans in model_answers:
                        for ref_ans in reference_answers:
                            sample = SingleTurnSample(
                                user_input=row['question'],
                                response=model_ans,
                                reference=ref_ans
                            )
                            score = await scorer.single_turn_ascore(sample)
                            max_score = max(max_score, score)
                    
                    score_list.append(max_score)
                else:
                    score_list.append(0.0)
                
            except Exception as e:
                print(f"处理第 {i+1} 个样本时出错: {e}")
                score_list.append(0.0)

        ragas_scored_dataset = input_dataset.add_column("ragas_AA_short", score_list)

        return ragas_scored_dataset

async def answer_accuracy_modified(input_dataset, evaluator_llm):
    # 在函数开始时创建一次 scorer
    scorer = AnswerAccuracy(llm=evaluator_llm)
    

    score_list = []
        
    for i, row in enumerate(tqdm(input_dataset, desc="Calculating short answer accuracy")):
        try:
            # 短答案评分 - 处理列表情况
            if 'short_answer' in row and 'model_new_answer' in row:
                model_answers = row['model_new_answer'] if isinstance(row['model_new_answer'], list) else [row['model_new_answer']]
                reference_answers = row['short_answer'] if isinstance(row['short_answer'], list) else [row['short_answer']]
                    
                # 计算所有组合的分数，取最高分
                max_score = 0.0
                for model_ans in model_answers:
                    for ref_ans in reference_answers:
                        sample = SingleTurnSample(
                                user_input=row['modified_question'],
                                response=model_ans,
                                reference=ref_ans
                            )
                        score = await scorer.single_turn_ascore(sample)
                        max_score = max(max_score, score)
                        if max_score == 1.0:
                            break  # 跳出内层循环
                    if max_score == 1.0:
                        break  # 跳出外层循环
                
                score_list.append(max_score)
            else:
                score_list.append(0.0)
                
        except Exception as e:
            print(f"处理第 {i+1} 个样本时出错: {e}")
            score_list.append(0.0)

    ragas_scored_dataset = input_dataset.add_column("new_AA", score_list)

    return ragas_scored_dataset

def batch_generate_responses_qwen3(tokenizer, model, prompts, system_prompt,
                             temperature=0.7, max_new_tokens=32768, batch_size=5,
                             enable_thinking=True, parse_thinking=True):
    """
    批量生成Qwen 3模型的回复

    参数:
        prompts: 字符串列表，每个字符串是一个提示词
        system_prompt: 系统提示词
        max_new_tokens: 生成的最大token数，默认为512
        batch_size: 每批处理的提示词数量，默认为4
        enable_thinking: 是否启用思考模式，默认为True
        parse_thinking: 是否解析思考内容，默认为True

    返回:
        如果parse_thinking=True，返回(思考内容列表, 回复内容列表)的元组
        如果parse_thinking=False，返回回复列表
    """

    responses = []
    thinking_contents = [] if parse_thinking else None

    # 按批次处理
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_texts = []

        # 为每个提示词准备输入文本
        for prompt in batch_prompts:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "user", "content": prompt}
                ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking  # 启用思考模式
            )
            batch_texts.append(text)

        # 批量编码输入
        model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.2
            )

        # 处理生成的回复
        batch_responses = []
        batch_thinking = [] if parse_thinking else None

        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
            response_ids = output_ids[len(input_ids):].tolist()

            if parse_thinking and enable_thinking:
                # 解析思考内容
                try:
                    # 寻找</think>对应的token ID (151668)
                    index = len(response_ids) - response_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                thinking_content = tokenizer.decode(response_ids[:index], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(response_ids[index:], skip_special_tokens=True).strip("\n")

                batch_thinking.append(thinking_content)
                batch_responses.append(content)
            else:
                # 不解析思考内容，直接返回完整回复
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                batch_responses.append(response)

        responses.extend(batch_responses)
        if parse_thinking:
            thinking_contents.extend(batch_thinking)

    if parse_thinking:
        return thinking_contents, responses
    else:
        return responses

def get_judgments_from_responses(responses: List[str]) -> List[Optional[str]]:
    """
    推荐的解析函数，结合了多种方法的优点
    """
    judgments = []

    for i, response in enumerate(responses):
        judgment = None

        try:
            # 方法1: 尝试直接解析JSON
            clean_response = response.strip()
            if clean_response.startswith('{') and clean_response.endswith('}'):
                data = json.loads(clean_response)
                judgment = data.get('judgment')
            else:
                # 方法2: 寻找JSON部分
                start_idx = clean_response.find('{')
                end_idx = clean_response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_part = clean_response[start_idx:end_idx+1]
                    data = json.loads(json_part)
                    judgment = data.get('judgment')

        except json.JSONDecodeError:
            # 方法3: 使用正则表达式作为后备
            pattern = r'"judgment"\s*:\s*"([^"]*)"'
            match = re.search(pattern, response)
            if match:
                judgment = match.group(1)

        if judgment is None:
            print(f"Warning: response {i} cannot retrieve judgment")
            print(f"Respond content: {response[:200]}...")
            judgment = "error"

        judgments.append(judgment)

    return judgments



def run_experiment(tokenizer, model, input_prompts, system_prompt, test_df):
  output = batch_generate_responses_qwen3(tokenizer, model, input_prompts, system_prompt)
  df = test_df.copy()
  df['MODIFIED_thinking'] = output[0]
  df['MODIFIED_model_response'] = output[1]
  to_process = output[1]
  processed_judgments = get_judgments_from_responses(to_process)
  df['MODIFIED_model_pred'] = processed_judgments

  return df

def prepare_test_prompts(test_df, task_text):
    print("Start preparing prompts...")
    base_prompt = task_text

    print(f"# Testing data points: {len(test_df)}")
    test_prompts = []
    for _, row in test_df.iterrows():
        query = row['modified_question']
        complete_prompt = base_prompt.replace("TARGET", query)
        test_prompts.append(complete_prompt)

    print(f"Generation complete: {len(test_prompts)} prompts")

    avg_length = sum(len(p) for p in test_prompts) // len(test_prompts)
    print(f"Average prompt length: {avg_length:,} bytes (~{avg_length//4:,} tokens)")

    return test_prompts