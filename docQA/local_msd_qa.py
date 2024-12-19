from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
# from configs.model_config import *
import datetime
from loc_qa import ChineseTextSplitter
from typing import List, Tuple
from langchain.docstore.document import Document
import numpy as np
# from utils import torch_gc
import os
import time
import torch

def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("If you are using macOS, it is recommended to upgrade PyTorch to version 2.0.0 or higher to support timely cleaning of memory usage generated by torch.")


VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content")

# LLM streaming response
STREAMING = True

# Context length after matching
CHUNK_SIZE = 250

# LLM input history length
LLM_HISTORY_LEN = 3

# Return top-k text chunks from vector store
VECTOR_SEARCH_TOP_K = 5

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"


# Prompt template based on context, please ensure to keep "{question}" and "{context}"
PROMPT_TEMPLATE = """Known Information:
{context} 

Based on the above known information, answer the user's question concisely and professionally. If you cannot derive an answer from it, please say “Cannot answer the question based on the known information” or “Insufficient relevant information provided.” Do not add fabricated content to the answer. Please use Chinese in the answer. The question is: {question}"""

def load_file(filepath):
    if filepath.lower().endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.lower().endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        textsplitter = ChineseTextSplitter(pdf=True)
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = ChineseTextSplitter(pdf=False)
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


def generate_prompt(related_docs: List[str],
                    query: str,
                    prompt_template=PROMPT_TEMPLATE) -> str:
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = prompt_template.replace("{question}", query).replace("{context}", context)
    return prompt


def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def separate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4,
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    docs = []
    id_set = set()
    store_len = len(self.index_to_docstore_id)
    for j, i in enumerate(indices[0]):
        if i == -1:
            # This happens when not enough docs are returned.
            continue
        _id = self.index_to_docstore_id[i]
        doc = self.docstore.search(_id)
        id_set.add(i)
        docs_len = len(doc.page_content)
        for k in range(1, max(i, store_len - i)):
            break_flag = False
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        break_flag = True
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
            if break_flag:
                break
    id_list = sorted(list(id_set))
    id_lists = separate_list(id_list)
    for id_seq in id_lists:
        for id in id_seq:
            if id == id_seq[0]:
                _id = self.index_to_docstore_id[id]
                doc = self.docstore.search(_id)
            else:
                _id0 = self.index_to_docstore_id[id]
                doc0 = self.docstore.search(_id0)
                doc.page_content += doc0.page_content
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        docs.append((doc, scores[0][j]))
    torch_gc()
    return docs


class LocalDocQA:
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                embedding_model: str = EMBEDDING_MODEL,
                embedding_device='cpu',
                top_k=VECTOR_SEARCH_TOP_K,
                ):

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model])
        # ,
                                                # model_kwargs={'device': embedding_device})
        self.top_k = top_k

    def init_knowledge_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None):
        loaded_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("Path does not exist.")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath)
                    print(f"{file} loaded successfully.")
                    loaded_files.append(filepath)
                except Exception as e:
                    print(e)
                    print(f"{file} failed to load.")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath)
                        print(f"{file} loaded successfully.")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} failed to load.")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_file(file)
                    print(f"{file} loaded successfully.")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} failed to load.")
        if len(docs) > 0:
            start = time.time()
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                vector_store.add_documents(docs)
                torch_gc()
            else:
                if not vs_path:
                    vs_path = os.path.join(VS_ROOT_PATH,
                                           f"""{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}""")
                
                vector_store = FAISS.from_documents(docs, self.embeddings)
                torch_gc()

            vector_store.save_local(vs_path)
            # Record program end time
            end = time.time()

            # Calculate program run time and output the result
            run_time = end - start
            print("Program run time: %.6f seconds" % run_time)
            return vs_path, loaded_files
        else:
            print("None of the files were successfully loaded. Please check the dependencies or replace with other files and upload again.")
            return None, loaded_files

    def get_knowledge_based_answer(self,
                                   query,
                                   vs_path,
                                   chat_history=[],
                                   streaming: bool = STREAMING):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        related_docs_with_score = vector_store.similarity_search_with_score(query,
                                                                            k=self.top_k)
        related_docs = get_docs_with_score(related_docs_with_score)
        torch_gc()
        return related_docs
        # yield related_docs
        # torch_gc()
        # prompt = generate_prompt(related_docs, query)

        # if streaming:
        #     for result, history in self.llm._stream_call(prompt=prompt,
        #                                                  history=chat_history):
        #         history[-1][0] = query
        #         response = {"query": query,
        #                     "result": result,
        #                     "source_documents": related_docs}
        #         yield response, history
        # else:
        # for result, history in self.llm._call(prompt=prompt,
        #                                       history=chat_history,
        #                                       streaming=streaming):
        #     torch_gc()
        #     history[-1][0] = query
        #     response = {"query": query,
        #                 "result": result,
        #                 "source_documents": related_docs}
        # yield response, history
            # torch_gc()


if __name__ == "__main__":

    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(
                          embedding_model=EMBEDDING_MODEL,
                          embedding_device='cpu',
                          top_k=VECTOR_SEARCH_TOP_K)
    # vs_path, _ = local_doc_qa.init_knowledge_vector_store('./MSD/disease_info.txt','docbase/MSD')
    # print(vs_path)
    # query = "The patient's question is about the diagnostic methods for pleural effusion and asks whether it can be diagnosed solely through an X-ray. The full name of pleural effusion is pleural effusion (pleural effusion)."
    query = "Can COPD patients drink alcohol?"
    last_print_len = 0
    k_neigbour = local_doc_qa.get_knowledge_based_answer(query=query,
                                                                 vs_path='./docbase/MSD',
                                                                 chat_history=[], 
                                                                 streaming=True)
    source_text = [f"""Source [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}:\n\n{doc.page_content}\n\n"""
                   f"""Relevance: {doc.metadata['score']}\n\n"""
                   for inum, doc in
                   enumerate(k_neigbour)]
    print("\n\n" + "\n\n".join(source_text))
    pass
