import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers.document_compressors import LLMChainExtractor
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from pathlib import Path


def string_to_rounded_list(s):
    s = s.strip('[]')
    return [round(float(x), 2) for x in s.split()] 

def text_query_generator(df,start_row,end_row):
    for i in range(start_row, end_row):
        post = df.iloc[[i]].reset_index(drop=True) 
        bd_risk = post['prob'][0] 
        future_su_30 = post['next_30_day_suicidaity'][0].lower() 
        query = f"""mental illness: {bd_risk} (MDD, BD)
posts:
"""
        for k in range(len(post['time'][0])):
            timestamp = post['time'][0][k] 
            date_string = timestamp.strftime('%m/%d/%Y')
            summary_suicidality = post['pred_pre'][0][k] 
            cur_su = post['past_6_month_suicidality'][0][k].lower()
            query += f"""- p{k+1}: {date_string}, {cur_su}, {summary_suicidality}
"""

        query += f"""-suicide_risk: {future_su_30}"""
    return query

def prompt_generator(df,start_row,end_row): 
    for i in range(start_row, end_row):
        post = df.iloc[[i]].reset_index(drop=True) 
        bd_risk = post['prob'][0]

        future_su_30 = post['next_30_day_suicidaity'][0].lower() 
        query = f"""mental illness: {bd_risk} (MDD, BD)
posts:
"""
        for k in range(len(post['time'][0])): 
            timestamp = post['time'][0][k] 
            date_string = timestamp.strftime('%m/%d/%Y')
            summary_suicidality = post['pred_pre'][0][k] 
            cur_su = post['past_6_month_suicidality'][0][k].lower() 
            query += f"""- p{k+1}: {date_string}, {cur_su}, {summary_suicidality}
"""
    return query


## 1. pre-process
def main(config):
    token_lim = config['token_limit']
    
    if config['model'] =='vitorcalvi/mentallama2':
        model = 'mentallama'
    elif config['model'] == 'phi3:latest':
        model = 'phi-3'

    result_save = config['result_save']
    
    df =pd.read_json('~/hyolim/JMIR_2024/_Data/240719_data/240803_df_embedding_processed_concat_8736.json')
    output_dict = {}
    df['prob'] = df['prob'].apply(string_to_rounded_list)
    df['time'] = df['time'].apply(lambda x: pd.to_datetime(x, unit='ms'))

    ## 2. RAG-Database
    database = df[df['train_test']=='rag'].reset_index(drop=True)
    embeddings_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/nli-roberta-large', 
        model_kwargs={'device':'cuda'},
        encode_kwargs={'normalize_embeddings':True})
    vectorstore = FAISS.load_local('vectorstore_SBERT/', embeddings = embeddings_model, allow_dangerous_deserialization = True) # 

    template = """
[instruction]
You are a psychiatrist. Given <input: posts> of in the past 6 months, predict <output: suicide_risk> in the next 30 days with.
Choose one of [suicide risk].

[suicide_risk]
1. indicator: never mention direct suicide thoughts, but share personal history of at-risk 
2. ideation: thought of suicide
3. behavior: actions planning for suicide (e.g. self-harm)
4. attempt: suicide attempt recently or indicate last wills

[similar cases]
{context}

[task]
<intput>
{question}
<output>
-suicide_risk: Choose one of [indicator, ideation, behavior, attempt]
"""

    prompt = ChatPromptTemplate.from_template(template=template)
    llm = ChatOllama(
        model=config['model'],  
        temperature=0,
        max_tokens=config['max_tok'], 
        model_kwargs={'seed':config['random_state']})

    retriever = vectorstore.as_retriever(search_type='mmr', 
        search_kwargs={'k': 3, 'fetch_k': config['result_save']})

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever= retriever, llm=llm)
    
    def format_docs(docs): 
        return '\n\n'.join([d.page_content for d in docs])
    
    chain = ({'context': multi_query_retriever|format_docs, 'question' : RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser())

    def pred_pred(inference, start, end):
        for i in tqdm(range(start, end)):
            query = prompt_generator(inference, i, i+1)  
            if len(query.split())+100 > config['token_limit']:
                inference['model_pred'][i] = 'token_limit'
            else:   
                docs = retriever.get_relevant_documents(query)  
                limit = config['token_limit'] -len(query.split()) - 100 
                doc_content = ''.join([doc.page_content for doc in docs])
                skip = 3

                while len(doc_content.split()) > limit:
                    longest_doc = max(docs, key=lambda doc: len(doc.page_content.split()))
                    docs.remove(longest_doc) 
                    additional_docs = retriever.get_relevant_documents(query, skip=skip) 
                    skip += 1  
                    
                    if skip > 30:
                        inference['model_pred'][i] = 'token_limit'
                        break
                        
                    if additional_docs:
                        docs.append(additional_docs[0])  
                    else:
                        break  
                    print(skip)

                    
                    doc_content = ''.join([doc.page_content for doc in docs])
                if skip<=30:
                    inference['docs'][i] = doc_content  
                    response = chain.invoke(doc_content)
                    inference['model_pred'][i] = response  
                else:
                    continue

        return inference

    df = df.reset_index(drop=True)
    df = df[df['train_test']=='test'].reset_index(drop=True)
    df['model_pred'] = ''
    df['docs']=''
    df_make = pred_pred(df, 0, len(df))

    # Result Save
    save_time = datetime.now().__format__("%m%d_%H%M%S%Z") #$ 
    save_path = f"/home/dsail/hyolim/JMIR_2024/_Baseline/suicide_risk/ollama/_Result/rag/{result_save}/{model}/"
    Path(f"{save_path}").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(df_make).to_json(f'{save_path}/{save_time}_{model}_rag_{result_save}.json')  

if __name__=="__main__":
    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",type=str, default='phi3:latest')#'vitorcalvi/mentallama2') # 
    parser.add_argument("--token_limit", type=int, default=4096)# -> 3910 / 2048 ->1860 / 4000 -> 3820
    parser.add_argument("--random_state", type=int, default=2023)#
    parser.add_argument("--max_tok", type=int, default=500) # llm inference length
    parser.add_argument("--result_save", type=int, default=30)# zeroshot, rag, fewshot
    config = parser.parse_args()
    print(config)

    main(config.__dict__)