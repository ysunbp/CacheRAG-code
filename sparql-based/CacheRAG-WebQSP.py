import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random
import ast
import json
import pickle
import numpy as np
from tqdm import tqdm, trange
import torch
from typing import List, Optional
from llama import Llama, Dialog
import json
import csv
import wikipedia
import requests
import pickle
import time
from transformers import (
    AutoTokenizer,
    pipeline, AutoModelForCausalLM
)
from examples import EXAMPLES

from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from test_dpr import DPRRetriever
from openai import APIConnectionError, OpenAI, RateLimitError
from loguru import logger

from rank_bm25 import BM25Okapi

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()


retriever = DPRRetriever(question_model, question_tokenizer, context_model, context_tokenizer)



img_suffix = ['jpg', 'png', 'svg', 'gif']

IN_CONTEXT_TRUE = [["Bangladesh Nationalist Party is the member of which international organization?", "according to the wikipedia page, bangladesh nationalist party is a member of the centrist democrat international.", "Asia Pacific Democrat Union or Centrist Democrat International"],
              ["What patrol aircraft is used by the South African Air Force?", "according to the wikipedia infobox, the patrol aircraft used by the south african air force is the c-47tp.", "C-47 Skytrain"],
              ["What party was split from Communist Refoundation Party?", "according to the wikipedia infobox, the italian communist party (pci) was split from to form the communist refoundation party (prc) in 1991.", "Italian Communist Party"],
              ["What is the stadium where BSG Chemie Leipzig (1950)'s home matches are held?", "alfred-kunze-sportpark (also known as georg-schwarz-sportpark)", "Alfred-Kunze-Sportpark or Georg-Schwarz-Sportpark"],
              ["What is the ending theme of My Papa Pi?", 'the ending theme of my papa pi is "my papa pi" by piolo pascual, pia wurtzbach, and pepe herrera.', "Pia Wurtzbach"],
              ["What is the legislative body in Albanian Kingdom (1928–1939)?", "according to the wikipedia infobox and summary, the legislative body in the albanian kingdom (1928–1939) is the constitutional assembly.", "Parliament of Albania"],
              ["The predecessor of Cirilo Almario is?", "manuel p. del rosario, d.d.", "Manuel del Rosario"],
              ["What is the mouth of Montreal River (Timiskaming District)?", "according to the wikipedia infobox and summary, the mouth of the montreal river (timiskaming district) is lake timiskaming on the ottawa river.", "Timiskaming, Unorganized, West Part, Ontario"],
              ["What significant design was created by Joseph Berlin?", "mograbi cinema, tel aviv.", "Cinema of Israel"],
              ["What patrol aircraft is used by the VPB-127?", "pv-1", "Lockheed Ventura or PV-1"]
              ]

IN_CONTEXT_FALSE = [["What/who influenced Charles Fourier?", "bob black.", "Nicolas-Edme Rétif"],
                    ["Which automobile team had the fastest driver during the 1960 Indianapolis 500?", "ken-paul", "A.J. Watson"],
                    ["Which company owns TV Land?", "paramount global.", "Paramount Media Networks"],
                    ["What language is spoken in Evening (magazine)?", "english", "Japanese language"],
                    ["What is the record label for Cogumelo Records?", "cogumelo records.", "Relapse Records"],
                    ["Jim Pearson was born in which place?", "chatham, ontario, canada.", "Falkirk"],
                    ["What is the format of The Wedge (Australian TV series)?", "the format of the wedge (australian tv series) is a sketch show.", "Stereophonic sound"],
                    ["Who developed Flappy?", "flappy bird was developed by .gears, which is a game development company founded by dong nguyen.", "DB-SOFT"],
                    ["What is Cinematic soul derived from?", "soul music, psychedelic soul, orchestral music, and film score.", "Disco"],
                    ["Which automobile team had the fastest driver during the 1953 Curtis Trophy?", "cooper-bristol.", "Cooper Car Company"]
                ]

cachefn = "./WebQSP/cache.jsonl"
cache = {}
cache["search"] = {}
cache["summary"] = {}
cache["page"] = {}
cache["html"] = {}
cache["infobox"] = {}
cache["sparql"] = {}
cache["sparql-one-hop"] = {}
cache["name2id"] = {}
cache["id2name"] = {}
cache["sparql-query"] = {}

with open(cachefn, "a", encoding='utf8') as f:
    pass

with open(cachefn, "r", encoding='utf8') as f:
    l = f.readline()
    while l:
        l = json.loads(l)
        if l[0] == 'sparql':
            cache[l[0]][l[1]+'-'+l[2]] = l[3]
        else:
            cache[l[0]][l[1]] = l[2]
        l = f.readline()

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

def get_wikidata_name(wikidata_id):
    global cache
    if wikidata_id in cache["id2name"]:
        return cache["id2name"][wikidata_id]
    else:
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={wikidata_id}&format=json"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            if 'entities' in data and wikidata_id in data['entities']:
                entity = data['entities'][wikidata_id]
                name = entity.get('labels', {}).get('en', {}).get('value', 'Name not found')
                with open(cachefn, "a", encoding='utf8') as f:
                    record = ["id2name", wikidata_id, name]
                    f.write(json.dumps(record) + "\n")
                return name
            else:
                return 'ID not found'
        else:
            return 'Error fetching data'

def get_wikidata_id(entity_name):
    global cache
    if entity_name in cache["name2id"]:
        return cache["name2id"][entity_name]
    else:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": entity_name,
            "language": "en",
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['search']:
            # 返回第一个匹配的实体ID
            with open(cachefn, "a", encoding='utf8') as f:
                record = ["name2id", entity_name, data['search'][0]['id']]
                f.write(json.dumps(record) + "\n")
            return data['search'][0]['id']
        else:
            return None

def llama_answer(
    dialog, generator,
    temperature = 0,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    

    dialogs: List[Dialog] = [
        [ {"role": "system", "content": dialog['system']},
            {"role": "user", "content": dialog['user']}]]  
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    output = []
    for dialog, result in zip(dialogs, results):
        output.append(result['generation']['content'])
    return output

def get_sparql_answer(cur_entity_relation):
    global cache, sparql
    cur_entity, cur_relation = cur_entity_relation
    if cur_entity and cur_relation:
        if cur_entity+'-'+cur_relation in cache['sparql']:
            return cache['sparql'][cur_entity+'-'+cur_relation]
            
        else:
            subject_entity = get_wikidata_id(cur_entity)
            
            query = f"""
            SELECT ?value WHERE {{
                wd:{subject_entity} wdt:{cur_relation} ?value.
            }}
            """
            #print(query)
            # 设置查询参数
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)

            # 执行查询
            results = sparql.query().convert()

            # 提取结果
            values = [result['value']['value'] for result in results['results']['bindings']]
            cur_result = []
            for value in values:
                if "http" in value:
                    cur_entity_name = get_wikidata_name(value.split("/")[-1])
                    if not (cur_entity_name == 'ID not found' or cur_entity_name == 'Error fetching data'):
                        cur_result.append(cur_entity_name)
                    else:
                        continue
                else:
                    cur_result.append(value)
            values = cur_result
            with open(cachefn, "a", encoding='utf8') as f:
                record = ["sparql", cur_entity, cur_relation, values]
                f.write(json.dumps(record) + "\n")
            return values
            
    else:
        return []

def get_sparql_query_answer(cur_sparql_query):
    global cache, sparql
    #cur_sparql_query = "SELECT DISTINCT ?x WHERE { wd:water wdt:melting_point ?x. }"
    if cur_sparql_query:
        if cur_sparql_query in cache['sparql-query']:
            print("reuse sparql query")
            return cache['sparql-query'][cur_sparql_query]
        else:
            try:
                query = cur_sparql_query
                #print(query)
                # 设置查询参数
                sparql.setQuery(query)
                sparql.setReturnFormat(JSON)

                # 执行查询
                results = sparql.query().convert()

                # 提取结果
                values = [result['x']['value'] for result in results['results']['bindings']]
                cur_result = []
                for value in values:
                    if "http" in value:
                        cur_entity_name = get_wikidata_name(value.split("/")[-1])
                        if not (cur_entity_name == 'ID not found' or cur_entity_name == 'Error fetching data'):
                            cur_result.append(cur_entity_name)
                        else:
                            continue
                    else:
                        cur_result.append(value)
                values = cur_result
                with open(cachefn, "a", encoding='utf8') as f:
                    record = ["sparql-query", cur_sparql_query, values]
                    f.write(json.dumps(record) + "\n")
                return values
            except:
                return []
            
    else:
        return []

def get_sparql_one_hop(cur_entity):
    global sparql, cache
    #cur_entity = "http://dbpedia.org/resource/Royal_Lao_Air_Force"
    cur_entity = get_wikidata_id(cur_entity)
    if cur_entity in cache["sparql-one-hop"]:
        return cache["sparql-one-hop"][cur_entity]
    else:
        result_triples = []
        try:
            query = f"""
            SELECT distinct ?predicate ?neighbor WHERE {{
                wd:{cur_entity} ?predicate ?neighbor.
                FILTER(?predicate != rdf:type)
            }}
            """
            #print(query)
            #query_core = "{entity} {relation} ?c".format(entity=dbpedia_entity, relation=dbpedia_relation)
            # 编写 SPARQL 查询
            #query = "select distinct ?c where{"+query_core+"}"
            #print(query)
            # 设置查询并执行
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            #results = g.query(query)
            results = sparql.query().convert()
            #print("RESULTS", results)

            # 提取结果
            triples = [(cur_entity, result['predicate']['value'], result['neighbor']['value']) for result in results['results']['bindings']]
            
            cur_topic_entity_name = get_wikidata_name(cur_entity)
            for cur_triple in triples:
                cur_result = [cur_topic_entity_name]
                cur_tail = cur_triple[1].split("/")[-1]
                if cur_tail in link2relation_dict:
                    cur_predicate = link2relation_dict[cur_tail]
                    cur_result.append(cur_predicate)
                else:
                    continue
                if "http" in cur_triple[-1]:
                    cur_entity_name = get_wikidata_name(cur_triple[-1].split("/")[-1])
                    if not (cur_entity_name == 'ID not found' or cur_entity_name == 'Error fetching data'):
                        cur_result.append(cur_entity_name)
                    else:
                        continue
                else:
                    cur_result.append(cur_triple[-1])
                print(cur_result)
                result_triples.append(cur_result)
            
            with open(cachefn, "a", encoding='utf8') as f:
                record = ["sparql-one-hop", cur_entity, result_triples]
                f.write(json.dumps(record) + "\n")
            return result_triples
        except:
            with open(cachefn, "a", encoding='utf8') as f:
                record = ["sparql-one-hop", cur_entity, result_triples]
                f.write(json.dumps(record) + "\n")
            return result_triples

def fuzz_match_partio(x, y):
    m = {}
    dist = -len(x)

    def f(i, j):
        if (i, j) in m:
            return m[(i, j)]
        if i == len(x):
            m[(i, j)] = 0
            return m[(i, j)]
        if j == len(y):
            m[(i, j)] = i - len(x)
            return m[(i, j)]
        m[(i, j)] = -len(x)
        if x[i] == y[j]:
            m[(i, j)] = max(m[(i, j)], f(i + 1, j + 1) + 1)
        m[(i, j)] = max(m[(i, j)], f(i, j + 1) - 1)
        m[(i, j)] = max(m[(i, j)], f(i + 1, j) - 1)
        return m[(i, j)]

    for i in range(len(y) - 1, -1, -1):
        dist = max(dist, f(0, i))
    return dist / len(x) if len(x) > 0 else 1

def get_relation(s, query):
    query = query.lower().replace(" ", "")
    bestrel = None
    bestscore = -1
    for i in range(len(s)):
        score = fuzz_match_partio(s[i].lower(), query)
        if score > bestscore or (score == bestscore and bestrel is not None and len(s[i]) > len(bestrel)):
            bestrel = s[i]
            bestscore = score
    return bestrel, bestscore

def compose_entity_template(cur_question):
    template = {}
    template['system'] = "You are a professional semantic parsing expert. Please provide your answer by giving brief (entity, predicate) pair. If no appropriate (entity, predicate) pair can be extracted, please respond (No, No)"
    template['user'] = "[Instruction]: The task is to determine the most appropriate (entity, predicate) pair based on a given question. MUST NOT provide answer to the question directly. "+'\n'+"[EXAMPLE]: Question: What aircraft bomber was used by the South Vietnam Air Force? '; Your Answer: ('South_Vietnam_Air_Force', 'aircraftBomber')"+'\n'+"[TASK]: Select the most appropriate (entity, predicate) based on this question and entity pair list: Question: "+cur_question
    return template

def compose_example(cur_tuple, gt):
    return '[EXAMPLE]: QUESTION: ' + cur_tuple[0] + '\n' + 'GROUND_TRUTH: ' + cur_tuple[2] + '\n' + 'ANSWER: ' + cur_tuple[1] + '\n' + 'Your answer: '+ gt + '\n'

def compose_hop_template(question, kb_content):
    dialog = {}
    dialog['system'] = "You will be given a QUESTION and a set of retrieved TRIPLEs from Wikidata. Your task is to indicate whether the currently retrieved content is sufficient for you to answer the QUESTION. If you need to have more retrieved triples, respond <NO>. If you think the subject entity is wrongly linked, respond <NA>. Otherwise, if you think the currect information is sufficient, respond <YES>. Only answer <NO>/<NA>/<YES>!!!"
    dialog["user"] = "QUESTION: "+question
    dialog["user"] += "Wikidata TRIPLEs: "
    dialog["user"] += str(kb_content)
    return dialog


def kb_one_hop_qa(question, naive_triples, generator):
    dialog = {}
    dialog['system'] = 'Please provide a brief answer as short as possible to the question based on your own knowledge and the following relevant TRIPLEs (subject, predicate, object) from Wikidata. Answer "I don\'t know" if you are not confident of your answer.'
    dialog['user'] = ""
    for cur_triple in naive_triples:
        dialog['user'] += 'TRIPLE: '
        dialog['user'] += str(cur_triple)
        dialog['user'] += '\n'
    dialog['user'] += question
    llama_current_answer = llama_answer(dialog, generator, temperature = 0)[0].strip().lower()
    print(dialog['user'])
    return llama_current_answer

def read_properties(relation_path):
    with open(relation_path, 'r') as file:
        data = json.load(file)
    property_dict = {}
    link_dict = {}
    for item in data:
        property_name = item["propertyLabel"]
        property_link = item["property"]
        property_dict[property_name] = property_link.split("/")[-1]
        link_dict[property_link.split("/")[-1]] = property_name
    return property_dict, link_dict

def compose_eval_template(question, ground_truth, answer):
    global IN_CONTEXT_TRUE, IN_CONTEXT_FALSE
    dialog = {}
    example = 'Here are some examples:' + '\n'
    few_shot = ""
    for i in range(len(IN_CONTEXT_TRUE)):
        few_shot += compose_example(IN_CONTEXT_TRUE[i], "Yes")
        few_shot += compose_example(IN_CONTEXT_FALSE[i], "No")
    dialog['system'] = 'The task is provided a QUESTION with GROUND_TRUTH answer, evaluate whether my ANSWER is correct, answer briefly with Yes/No. You will first see some [EXAMPLE]s on this task and then you will complete the [TASK].'
    dialog['user'] = example+few_shot+'[TASK]: QUESTION: ' + str(question) + '\n' + 'GROUND_TRUTH: ' + str(ground_truth) + '\n' + 'ANSWER: ' + str(answer) + '\n' + "Your answer is?"
    return dialog

def move_tensors_to_gpu(data):
    for key, value in data.items():
        if torch.is_tensor(value):
            data[key] = value.to('cuda')
    return data

def wikisp(query, entity, wikisp_generator, wikisp_tokenizer, flag = 1):
    if flag:
        _instruction = "Given a query with resolved entities, generate the corresponding SPARQL. Use property names instead of PIDs."
        _input = "Query: {}\nEntities: {};".format(query, entity)
        inputs = "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:".format(_instruction, _input)
    else:
        _instruction = "Given a query, generate the corresponding SPARQL. Use property names instead of PIDs."
        _input = "Query: {}".format(query)
        inputs = "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:".format(_instruction, _input)
    print(inputs)
    tokenized_inputs = wikisp_tokenizer(inputs, return_tensors="pt")
    tokenized_inputs = move_tensors_to_gpu(tokenized_inputs)
    generate_ids = wikisp_generator.generate(tokenized_inputs.input_ids, max_length=200)
    answer = wikisp_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return str(answer).split(inputs)[-1].split("</s>")[0]

import re
def extract_entities_relations(sparql_query):
    # 定义正则表达式来匹配实体
    entity_pattern = r'wd:([\w_]+)'  # 匹配以 wd: 开头的实体
    #variable_pattern = r'\?([\w_]+)'  # 匹配变量，如 ?x
    relation_pattern = r'wdt:([\w_]+)'

    # 查找所有实体和变量
    entities = re.findall(entity_pattern, sparql_query)
    relations = re.findall(relation_pattern, sparql_query)
    #variables = re.findall(variable_pattern, sparql_query)

    # 合并实体和变量
    return entities, relations

def attempt_api_call(client, model_name, messages, max_retries=3):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            print(messages)
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
            )
            #print(response)
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def store_example_caching_by_content(example_cache_content, record):
    # record composes of question:thoughts-api function call
    query, plan = record[1], record[2]
    example_cache_content[query] = plan
    return example_cache_content

def store_neighbor_caching_by_content(neighbor_cache_content, neighbor_cache_retrieval, record):
    # record composes of question:thoughts-api function call
    query, retrieved_content, plan = record[0], record[1], record[2]
    neighbor_cache_content[query] = plan
    neighbor_cache_retrieval[query] = retrieved_content
    return neighbor_cache_content, neighbor_cache_retrieval

def generate_sparql_prompt(examples):
    result = ""
    questions, plans = examples
    for idx in range(len(plans)):
        result += "Question: "
        result += questions[idx]
        result += "\n"
        result += "Answer: "
        result += plans[idx]
        result += "\n"
        result += "\n"
    result += """Reminder:
    - SPARQL queries MUST follow the format if specified.
    - If there is no SPARQL query available, answer I don\'t know.'
    """
    return result

def generate_neighbor_reason_prompt(examples):
    result = ""
    questions, retrieved_content, plans = examples
    for idx in range(len(plans)):
        result += "Question: "
        result += questions[idx]
        result += "\n"
        result += "Retrieved content: "
        result += retrieved_content[idx]
        result += "\n"
        result += "Answer: "
        result += plans[idx]
        result += "\n"
        result += "\n"
    result += """Reminder:
    - MUST follow the format if specified.
    - Answer I don\'t know, if you are not sure about your answer.'
    """
    return result

def generate_meta_template(openai_client, cur_question, examples):
    cur_prompt = "Please compose proper SPARQL queries to retrieve relevant content to the question. You need to provide your thoughts on your choices. Start your thoughts with 'THOUGHTS: ', start with your answer with 'YOUR ANSWER: '."+generate_sparql_prompt(examples)
    print("current prompt",cur_prompt)
    cur_router_prompt = {}
    cur_router_prompt["system"] = "You are an expert in problem analysis and generalization. "+cur_prompt
    cur_router_prompt["user"] = "The current question: "+cur_question
    messages = [{"role":"system", "content": cur_router_prompt["system"]}, {"role":"user", "content": cur_router_prompt["user"]}]
    response = attempt_api_call(openai_client, "deepseek-chat", messages)
    return response

def get_example_neighbor_caching(query, neighbor_cache_content, neighbor_cache_retrieval, lambda_param=0, k=1):
    candidate_documents_pool = neighbor_cache_content
    candidate_query_pool = list(candidate_documents_pool.keys())
    
    bm25 = BM25Okapi([doc.split(" ") for doc in candidate_query_pool])
    scores = bm25.get_scores(query.split(" "))
    selected_docs = []

    for _ in range(min(k, len(candidate_query_pool))):
        # 计算每个文档的 MMR 分数
        mmr_scores = []
        for i in range(len(candidate_query_pool)):
            if i in selected_docs:
                continue
            
            relevance = scores[i]
            redundancy = 0.0
            
            # 计算冗余
            for selected in selected_docs:
                redundancy += bm25.get_scores(candidate_query_pool[selected].split(" "))[i]
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, i))
        
        # 选择最高 MMR 分数的文档
        best_doc = max(mmr_scores, key=lambda x: x[0])[1]
        selected_docs.append(best_doc)

    hit_queries = [candidate_query_pool[idx] for idx in selected_docs]
    hit_content = [neighbor_cache_retrieval[cur_query] for cur_query in hit_queries]
    hit_plans = [neighbor_cache_content[cur_query] for cur_query in hit_queries]

    return hit_queries, hit_content, hit_plans


def get_example_caching(query, example_cache_content, lambda_param=0, k=5):
    candidate_documents_pool = example_cache_content
    candidate_query_pool = list(candidate_documents_pool.keys())
    
    bm25 = BM25Okapi([doc.split(" ") for doc in candidate_query_pool])
    scores = bm25.get_scores(query.split(" "))
    selected_docs = []

    for _ in range(min(k, len(candidate_query_pool))):
        # 计算每个文档的 MMR 分数
        mmr_scores = []
        for i in range(len(candidate_query_pool)):
            if i in selected_docs:
                continue
            
            relevance = scores[i]
            redundancy = 0.0
            
            # 计算冗余
            for selected in selected_docs:
                redundancy += bm25.get_scores(candidate_query_pool[selected].split(" "))[i]
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((mmr_score, i))
        
        # 选择最高 MMR 分数的文档
        best_doc = max(mmr_scores, key=lambda x: x[0])[1]
        selected_docs.append(best_doc)

    hit_queries = [candidate_query_pool[idx] for idx in selected_docs]
    hit_plans = [example_cache_content[cur_query] for cur_query in hit_queries]

    return hit_queries, hit_plans

def compose_entity_selection(extracted_entities, cur_query, openai_client):
    cur_prompt = {}
    cur_prompt["system"] = """You are an expert in question analysis and answering. 
    Your task is to identify the key entity based on the given question. 
    We will then retrieve the one-hop KB neighbors based on the key entities selected so as to answer the given question.
    Return the key entities you have selected in a list, or returns an empty list (e.g., []) if you don't think there is a suitable key entity.
    Please only return the key entity list, without providing intermediate thinking processes!!!"""
    cur_prompt["user"] = "The current question: "+cur_query
    #cur_prompt["user"] += "\n"
    #cur_prompt["user"] += "The candidate entities: "+str(extracted_entities)
    messages = [{"role":"system", "content": cur_prompt["system"]}, {"role":"user", "content": cur_prompt["user"]}]
    response = attempt_api_call(openai_client, "deepseek-chat", messages)
    return response


def cacherag(question_set, wikisp_generator, wikisp_tokenizer, generator, out_path, relation2link_dict, openai_client, dpr_one_hop = 10):
    
    questions, wiki_answers = question_set
    random.seed(20)
    indices = random.sample(range(len(questions)), 125)
    questions = [questions[i] for i in indices]
    wiki_answers = [wiki_answers[i] for i in indices]
    
    neighbor_cache_content = {}
    neighbor_cache_retrieval = {}
    with open('generated_records.pkl', 'rb') as file:
        all_record = pickle.load(file)
    for item in all_record:
        cur_question, cur_triples, cur_answer = item
        neighbor_cache_content[cur_question] = cur_answer
        neighbor_cache_retrieval[cur_question] = cur_triples

    all_lambda = [0.5]
    for cur_lambda in all_lambda:
        example_cache_content = {} # 暂不考虑auto generation
        example_cache_content["After whom is the Riemannian geometry named?"] = "SELECT DISTINCT ?x WHERE { wd:riemannian_geometry wdt:named_after ?x. }"
        example_cache_content["What is the boiling point of water?"] = "SELECT DISTINCT ?x WHERE { wd:water wdt:boiling_point ?x. }"
        example_cache_content["At which school was Yayoi Kusama educated at?"] = "SELECT DISTINCT ?x WHERE { wd:yayoi_kusama wdt:educated_at ?x. }"

        total = 0
        correct = 0
        auto_eval_correct = 0
        auto_eval_incorrect = 0
        miss = 0
        sp_success = 0
        llm_success = 0
        neighbor_success = 0
        all_time = []

        with open(out_path, 'a', newline='') as outfile:
            for idx in trange(len(questions)):
                total_time = 0
                total += 1
                cur_answer = wiki_answers[idx]
                cur_answer_string = str(cur_answer)
                cur_question = questions[idx]
                print(cur_question)
                flag = 0
                main_entity = None
                t1 = time.time()
                wikisp_response = wikisp(cur_question, main_entity, wikisp_generator, wikisp_tokenizer, flag)
                
                original_wikisp_sparql = wikisp_response.split("Response:")[-1]
                print("sparql", original_wikisp_sparql)
                
                extracted_entities, extracted_relations = extract_entities_relations(original_wikisp_sparql)
                t2 = time.time()
                total_time += (t2-t1)
                print(extracted_entities, extracted_relations)
                extracted_entity_ids = []
                extracted_relation_ids = []
                for extracted_entity in extracted_entities:
                    extracted_entity_ids.append(get_wikidata_id(extracted_entity))
                for extracted_relation in extracted_relations:
                    cleaned_relation = extracted_relation.replace("_", " ")
                    if cleaned_relation in relation2link_dict:
                        relation_id = relation2link_dict[cleaned_relation]
                    else:
                        fuzzy_match = get_relation(list(relation2link_dict.keys()), cleaned_relation)[0]
                        relation_id = relation2link_dict[fuzzy_match]
                    extracted_relation_ids.append(relation_id)
                continue_flag = False

                for cur_entity_idx in range(len(extracted_entity_ids)):
                    if extracted_entity_ids[cur_entity_idx]:
                        wikisp_sparql = original_wikisp_sparql.replace(extracted_entities[cur_entity_idx], extracted_entity_ids[cur_entity_idx])
                    else:
                        continue_flag = True
                for cur_relation_idx in range(len(extracted_relation_ids)):
                    wikisp_sparql = original_wikisp_sparql.replace(extracted_relations[cur_relation_idx], extracted_relation_ids[cur_relation_idx])
                print("updated sparql", wikisp_sparql)

                if not continue_flag:
                    t3 = time.time()
                    sp_answers = get_sparql_query_answer(wikisp_sparql) #若直接有sp_answers就返回 没有的话找one hops
                    t4 = time.time()
                    if sp_answers:
                        total_time += (t4-t3)
                        sp_success += 1
                        cur_generated_record = ["empty", cur_question, wikisp_sparql]
                        example_cache_content = store_example_caching_by_content(example_cache_content, cur_generated_record)
                else:
                    sp_answers = []
                print("sp answers", sp_answers)
                
                if not sp_answers:
                    if neighbor_cache_content:
                        cur_examples = get_example_neighbor_caching(cur_question, neighbor_cache_content, neighbor_cache_retrieval, lambda_param=cur_lambda, k=1)
                        example_query, example_retrieval, example_answer = cur_examples
                        cur_examples_flag = False
                    else:
                        cur_examples_flag = True
                    # curexamples包含三个列表 queries, retrieved content, plans
                    
                    neighbor_success += 1
                    print(original_wikisp_sparql)
                    extracted_entities, _ = extract_entities_relations(original_wikisp_sparql)
                    extracted_entities = list(set(extracted_entities))
                    t9 = time.time()
                    selected_entities = compose_entity_selection(extracted_entities, cur_question, openai_client)
                    try:
                        selected_entities = ast.literal_eval(selected_entities)
                    except:
                        selected_entities = []
                    print(selected_entities)
                    if selected_entities:
                        all_content = []
                        for cur_entity in selected_entities:
                            
                            naive_triples = get_sparql_one_hop(cur_entity)
                            print("naive triples", naive_triples)
                            
                            all_triples = naive_triples
                            if len(all_triples) > dpr_one_hop:
                                naive_triples_string = [item[1].lower() for item in all_triples]
                                cur_detected_entity = get_wikidata_name(cur_entity)
                                cur_replaced_question = cur_question.lower().replace(cur_detected_entity.lower(), '').replace(cur_detected_entity.split('(')[0].lower(), '')
                                top_k_indices_and_scores = retriever.retrieve_top_k(cur_replaced_question, naive_triples_string, k=dpr_one_hop)
                                top_k_results = [all_triples[item] for item, score in top_k_indices_and_scores]
                                all_triples = top_k_results
                            naive_triples = all_triples
                            all_content += naive_triples

                        try:
                            if not cur_examples_flag:
                                messages = [{"role":"user", "content":"You are a professional question answering assistant, please provide brief answer to the QUESTION based on the given KB triple content, if you don't think the KB content contains the answer, answer the question based on your own knowledge! Here is an example: [EXAMPLE QUESTION]: "+example_query+"[EXAMPLE KB TRIPLES]: "+example_retrieval+"[EXAMPLE ANSWER]: "+example_answer+"\nNow answer the CURRENT QUESTION with CURRENT KB TRIPLES: "+"[CURRENT KB TRIPLES]: "+str(all_content)+"\n"+"[CURRENT QUESTION]: "+cur_question}]
                            else:
                                messages = [{"role":"user", "content":"You are a professional question answering assistant, please provide brief answer to the QUESTION based on the given KB triple content, if you don't think the KB content contains the answer, answer the question based on your own knowledge! "+"\nNow answer the CURRENT QUESTION with CURRENT KB TRIPLES: "+"[CURRENT KB TRIPLES]: "+str(all_content)+"\n"+"[CURRENT QUESTION]: "+cur_question}]
                            llm_answer = attempt_api_call(openai_client, "deepseek-chat", messages)
                            time.sleep(1)
                            record = [cur_question, str(all_content), llm_answer]
                            neighbor_cache_content, neighbor_cache_retrieval = store_neighbor_caching_by_content(neighbor_cache_content, neighbor_cache_retrieval, record)
                        except:
                            flag = True
                            tic = time.time()
                            while flag:
                                try:
                                    all_content = all_content[:int(len(all_content)/2)]
                                    if not cur_examples_flag:
                                        messages = [{"role":"user", "content":"You are a professional question answering assistant, please provide brief answer to the QUESTION based on your own knowledge or the given KB triple content. Here is an example: [EXAMPLE QUESTION]: "+example_query+"[EXAMPLE KB TRIPLES]: "+example_retrieval+"[EXAMPLE ANSWER]: "+example_answer+"\nNow answer the CURRENT QUESTION with CURRENT KB TRIPLES: "+"[CURRENT KB TRIPLES]: "+str(all_content)+"\n"+"[CURRENT QUESTION]: "+cur_question}]
                                    else:
                                        messages = [{"role":"user", "content":"You are a professional question answering assistant, please provide brief answer to the QUESTION based on your own knowledge or the given KB triple content. "+"\nNow answer the CURRENT QUESTION with CURRENT KB TRIPLES: "+"[CURRENT KB TRIPLES]: "+str(all_content)+"\n"+"[CURRENT QUESTION]: "+cur_question}]
                                    llm_answer = attempt_api_call(openai_client, "deepseek-chat", messages)
                                    time.sleep(1)
                                    record = [cur_question, str(all_content), llm_answer]
                                    neighbor_cache_content, neighbor_cache_retrieval = store_neighbor_caching_by_content(neighbor_cache_content, neighbor_cache_retrieval, record)
                                    flag = False
                                except:
                                    flag = True
                                    toc = time.time()
                                    if toc-tic > 60:
                                        llm_answer = "I don\'t know"
                                        break
                        if 'don\'t know' in llm_answer:
                            llm_success += 1
                            messages = [{"role":"user", "content":"You are a professional question answering assistant, please provide brief answer to the QUESTION based on your own knowledge."+"\n"+"QUESTION: "+cur_question}]
                            llm_answer = attempt_api_call(openai_client, "deepseek-chat", messages)
                            time.sleep(1)
                    else:
                        messages = [{"role":"user", "content":"You are a professional question answering assistant, please provide brief answer to the QUESTION based on your own knowledge."+"\n"+"QUESTION: "+cur_question}]
                        llm_answer = attempt_api_call(openai_client, "deepseek-chat", messages)
                        time.sleep(1)
                
                else:
                    llm_answer = str(sp_answers)[1:-1]
                t10 = time.time()
                total_time += t10-t9
                all_time.append(total_time)
                #cur_answer_string = cur_answer_string.split("<")[0]+"']"
                #cur_answer_string = cur_answer_string[2:-2]
                print('llama response', llm_answer)
                print('current gt', cur_answer_string)
                item = {}
                item["question"] = cur_question
                item["answer"] = cur_answer_string
                item["llm_answer"] = llm_answer
                json_line = json.dumps(item)
                outfile.write(json_line + '\n')
                if (cur_answer_string.lower() in llm_answer.lower()) or (cur_answer_string.split("(")[0].strip().lower() in llm_answer.lower()) or (cur_answer_string.lower() in llm_answer.replace('_', " ").lower()) or (cur_answer_string.replace("[","").replace("]","").replace("'","").lower() in llm_answer.lower()):
                    correct += 1
                    auto_eval_correct += 1
                else:
                    if 'don\'t know' in llm_answer:
                        miss += 1
                    else:
                        eval_template = compose_eval_template(cur_question, cur_answer_string.lower(), llm_answer)
                        try:
                            llama_current_answer = llama_answer(eval_template, generator, temperature = 0)[0].strip().lower()
                        except:
                            llama_current_answer = "yes"
                        if 'no' in llama_current_answer:
                            print('llama no eval', llama_current_answer)
                            auto_eval_incorrect += 1
                        else:
                            auto_eval_correct += 1
        print('+++++++++++++++++++++++++++++++')
        print('Now checking dpr hop', dpr_one_hop)
        print('total', total, 'correct', correct, 'miss', miss, 'accuracy', correct/total, 'miss rate', miss/total)
        print('auto eval accuracy', auto_eval_correct/total)
        print('hallucination rate', 1-miss/total-auto_eval_correct/total)
        print("trustfulness score", auto_eval_correct/total-(1-miss/total-auto_eval_correct/total))
        print("all times", all_time)
        print("mean time", np.mean(all_time))
        eval_result_dict = {}
        eval_result_dict['accuracy'] = auto_eval_correct/total
        eval_result_dict['missing'] = miss/total
        eval_result_dict['hallucination'] = 1-miss/total-auto_eval_correct/total
        eval_result_dict['score'] = auto_eval_correct/total-(1-miss/total-auto_eval_correct/total)
        eval_result_dict['total'] = total
        print(eval_result_dict)
        print(sp_success, llm_success, neighbor_success)


if __name__ == "__main__":
    openai_client = OpenAI(base_url='xxxxx', api_key="xxxxx")
    cur_model_dir = "./wikidata-emnlp23/trained_model"
    wikisp_tokenizer = AutoTokenizer.from_pretrained(cur_model_dir)
    wikisp_generator = AutoModelForCausalLM.from_pretrained(cur_model_dir).cuda()

    with open('./WebQSP/questions.pkl', 'rb') as file:
        questions = pickle.load(file)
    with open('./WebQSP/answers.pkl', 'rb') as file:
        answers = pickle.load(file)
    
    cur_model_dir = './llama3/Meta-Llama-3-8B-Instruct/'
    generator = Llama.build(
            ckpt_dir=cur_model_dir,
            tokenizer_path=cur_model_dir+'tokenizer.model',
            max_seq_len=8192,
            max_batch_size=6)
    
    question_set = [questions, answers]
    relation_path = "./WebQSP/properties.json"
    relation2link_dict, link2relation_dict = read_properties(relation_path)
    outpath = "./json/WebQSP-cacherag-autogen.json"
    cacherag(question_set, wikisp_generator, wikisp_tokenizer, generator, outpath, relation2link_dict, openai_client)

# python -m torch.distributed.run --nproc_per_node 1 