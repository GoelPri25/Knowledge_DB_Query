from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
from sentence_transformers import SentenceTransformer
import torch
import traceback

#from load_model import load_model
import re
from typing import List, Optional, Tuple, Union

import numpy as np

from transformers.pipelines.base import PIPELINE_INIT_ARGS, ChunkPipeline

import time 
import pathlib
import fitz

from transformers.utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_pytesseract_available,
    is_torch_available,
    is_vision_available,
    logging,
)

import pytesseract

from PIL import Image

if is_torch_available():
    import torch

    # We do not perform the check in this version of the pipeline code
    # from transformers.models.auto.modeling_auto import MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING
import json
import torch
print(torch.cuda.is_available())
logger = logging.get_logger(__name__)

class load_model():
    def __init__(self):

        #self.device = 0 if torch.cuda.is_available() else 1
        self.tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space = True)
        model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="ff904df")
        
        #self.model = model.to(self.device)
        self.model = model

        self.emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


class pageInfo():
    def __init__(self, words=[], boxes=[]):
        self.words = words
        self.boxes = boxes

    
class preprocessObjs():
    def __init__(self, file_info, query_info):
        self.file_info = file_info
        self.query_info = query_info
        
class queryDocService():
    
    def __init__(self, file_paths, queries):
        token_model = load_model()
        #self.device = token_model.device
        self.model = token_model.model
        self.emb_model = token_model.emb_model
        self.tokenizer = token_model.tokenizer
        self.file_paths = file_paths
        self.queries = queries
        start = time.time()
        self.processed_files = self.process_files()
      
        final = time.time() - start
        print("Time to preprocess files: ", final)
    
     
    def normalize_box(self, box, width, height):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / height)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / height)),
        ]

    def get_pageInfo(self, word_info, page_width, page_height):
        words = []
        boxes = []
        for each in word_info:
            try:
                words.append(each[4])
                boxes.append(self.normalize_box([each[0], each[1], each[2], each[3]], page_width, page_height))
            except:
                pass
        return pageInfo(words, boxes)


    def file_data(self, file_path):
        docInfo = []
        try:
            doc = fitz.open(file_path)
            count = 0
            for page in doc:
                try:
                    page_width = page.mediabox.width
                    page_height = page.mediabox.height

                    word_info = page.get_text("words")
                    page_info = self.get_pageInfo(word_info, page_width, page_height)

                    docInfo.append({'page_id': count,'words':page_info.words, 'boxes': page_info.boxes})

                except:
                    pass
                count = count + 1
        except:
            print(traceback.print_exc())
        return docInfo

    def encoding_bbox(self, encoding, boxes):
        
        bbox = []
        for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
            
            if s == 1:
                bbox.append(boxes[w])
            elif i == self.tokenizer.sep_token_id:
                bbox.append([1000] * 4)
            else:
                bbox.append([0] * 4)
        
       
        #encoding["bbox"] = torch.tensor([bbox]).to(self.device)
        encoding["bbox"] = torch.tensor([bbox])
        
        return encoding

    def get_bbox(self, start_bbox, end_bbox):
        try:
            bbox = [start_bbox[0], start_bbox[1], end_bbox[2], end_bbox[3]]

            return bbox
        except:
            print(traceback.print_exc())
            return None


    def decoder(self, outputs, word_ids, words, boxes):
      try:
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]


        words_ext = words[start : end + 1]

        temp = {}
        if len(words_ext) <= 6 :

          result = " ".join(words[start : end + 1])
          temp['text'] = result
          temp['bbox'] = self.get_bbox(boxes[start], boxes[end])

          return temp

      except:
        print(traceback.print_exc())
        return None




    def answer_pageInfo(self, question_split, words, boxes):

     
      encoding = self.tokenizer.encode_plus(
        question_split, words, is_split_into_words=True,  return_tensors="pt", stride = 128,
         truncation="only_second", padding = True
      )
      
      encoding = self.encoding_bbox(encoding, boxes)

      outputs = self.model(**encoding)
      
      word_ids = encoding.word_ids(0)
      answer = self.decoder(outputs, word_ids, words, boxes)
      
      
      return answer


    def eval_answer_doc(self, ques_split, doc_info):
        
        answer_doc = []
        count = 0
        for each_page in doc_info:
          try:

              answer = self.answer_pageInfo(ques_split, each_page['words'], each_page['boxes'])

              if answer is not None:
                temp = {}
                temp['page_id'] = each_page['page_id']
                #to be changed later
                temp['page_answers'] = [answer]
                answer_doc.append(temp)

              count = count + 1
              if count == 10:
                break

          except:
            pass

        return answer_doc
    
    
    def query_file(self, file_path, query):
        pass
    
    def process_files(self):
        file_infos = []
        
        for file_path in self.file_paths:
            try:
                temp = {}
                doc_info = self.file_data(file_path)
                temp['file_path'] = file_path
                temp['file_info'] = doc_info
                file_infos.append(temp)
            except:
                pass
            
        return file_infos
    
    def iterate_doc(self, query_split):
        all_docs = []
        count = 0
        
        for doc_info in self.processed_files:
            try:
                temp = {}

                answers_doc = self.eval_answer_doc(query_split, doc_info['file_info'])
                temp['doc_path'] = doc_info['file_path']
                temp['doc_id'] = count
                temp['doc_results'] = answers_doc
                all_docs.append(temp)
                count = count + 1 
            except:
                pass
            
        return all_docs

    def create_embs(self, sentences):
      
      embs = self.emb_model.encode(sentences)
      
      for each in embs[1:]:
        try:
          embs[0] = embs[0] + each
        except:
          pass

      return embs[0]
    
    

    def db_obj_result(self,info):
        try:
            temp = {}
            temp['file_info'] = info.file_info
            temp['query_info'] = info.query_info
            return temp
        except:
            print(traceback.print_exc())
            return None
    
    def ques_answer_service(self):
        
        for query_type, query_val in self.queries.items():
          try:
              query_split = query_val["queries"][0].split()
              doc_results = self.iterate_doc(query_split)
              self.queries[query_type]["answers"] = doc_results
              self.queries[query_type]['vector'] = self.create_embs(query_val["queries"]).tolist()
              #print(self.queries[query_type]['embedding'])

          except:
            pass
        try:
            result = self.db_obj_result(preprocessObjs(self.processed_files , self.queries))    
            return result
        except:
            print(traceback.print_exc())
            return None

    


if __name__ == "__main__":
    file_paths = ["/home/priyanjali/nlqir/10-K.pdf", "/home/priyanjali/nlqir/10-K-HSBC.pdf"]
    queries = {"headings":{"queries":["what are the headings in the given page?",
                                      "how many headings are available, extract",
                                      "titles in the pages of the document",
                                      "titles and headings of the doc"], 
                          "vector":None},"org":{"queries":["list the name of organisations in pdfs",
                                                          "list org in the pdf ",
                                                          "identify all the company names",
                                                          "find company name in docs"], "vector":None}}
    
    input_data = {'file_paths':file_paths, "questionList":queries}
    

   

    start = time.time()
    result_info = queryDocService(input_data['file_paths'], input_data['questionList']).ques_answer_service()
    
    final = time.time() - start
    

    print("Time taken to evaluate queries on docs: ", final)
    