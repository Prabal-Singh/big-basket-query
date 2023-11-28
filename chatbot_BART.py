from searcher import NeuralSearcher
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class ChatBot:
    def __init__(self, model_name, filename, indices_dict):
        self.model_name = model_name
        self.filename = filename
        self.indices_dict = indices_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def collect_context(self, query, indices_dict):
        searcher = NeuralSearcher(collection_name="Product Infos", model="bert-base-nli-mean-tokens")
        results = searcher.search(query, limit=10)
        indices = [indices_dict[i['sentence']] for i in results] 
        return indices

    def get_docs(self, question, filename, indices_dict):
        indices = self.collect_context(question, indices_dict) # gets the indices of the relevant context
        documents = []
        df = pd.read_csv(filename)
        for i in indices:
            row = df.iloc[i]
            context = f'''The Big Basket Store has the following product in its stock: Product Name is {row["product"]}, Category is {row["category"]}, Sub Category is {row["sub_category"]}, Brand is {row["brand"]}. It costs {row["sale_price"]} rupees, Rating is {row["rating"]} out of 5,
                            More about the product: {row["description"]}\n\n'''
            documents.append(context)
        conditioned_doc = "<P> " + " <P> ".join([d for d in documents])
        return conditioned_doc

    def get_answer(self, question, model, tokenizer, filename, indices_dict):
        conditioned_doc = self.get_docs(question, filename=filename, indices_dict=indices_dict)
        query_and_docs = "question: {} context : {}".format(question, conditioned_doc)
        model_input = tokenizer(query_and_docs, truncation=True, padding=True, return_tensors="pt")
        print(query_and_docs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generated_answers_encoded = model.generate(input_ids=model_input["input_ids"].to(device),
                                            attention_mask=model_input["attention_mask"].to(device),
                                            min_length=64,
                                            max_length=256,
                                            do_sample=False, 
                                            early_stopping=True,
                                            num_beams=8,
                                            temperature=1.0,
                                            top_k=None,
                                            top_p=None,
                                            eos_token_id=tokenizer.eos_token_id,
                                            no_repeat_ngram_size=3,
                                            num_return_sequences=1)
        return tokenizer.batch_decode(generated_answers_encoded, skip_special_tokens=True,clean_up_tokenization_spaces=True)[0]


    def get_response(self, prompt):
        if prompt is None:
            return "What is up?"
        response = self.get_answer(prompt, self.model, self.tokenizer, self.filename, self.indices_dict)
        if response is None:
            response = "What is up?"
        return response
