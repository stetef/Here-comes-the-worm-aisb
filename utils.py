import os
from dotenv import load_dotenv

import anthropic
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import consts

import evaluate
from tqdm import tqdm
import random
import json

load_dotenv()

def call_open_ai(prompt, model_name="gpt-4o-mini", 
                 temperature=1, max_tokens=256) -> str:
    gpt_client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY")
    )
    chat_completion = gpt_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        temperature=temperature,
        max_completion_tokens=max_tokens
    )
    return chat_completion.choices[0].message.content

def call_claude(prompt, model_name="claude-opus-4-1-20250805", 
                max_tokens=1024) -> str:
    claude_client = anthropic.Anthropic(
        api_key = os.getenv("ANTHROPIC_API_KEY")
    )
    message = claude_client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", 
             "content": prompt}
        ]
    )
    return message.content

def call_hf_model(prompt, model_name='Qwen/Qwen3-0.6B', 
                  system_prompt=None, 
                  max_new_tokens=32) -> str:

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"USING {device.type}")

    def get_hf_model_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer

    def query_hf_model(model, tokenizer, prompt, device, 
                       system_prompt=None, max_new_tokens=32):
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        return model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    def decode_outputs(outputs):
        parsed = tokenizer.decode(outputs[0])
        return parsed.split("**Answer:** ")[-1].replace("<|im_end|>", "")
    
    model, tokenizer = get_hf_model_tokenizer(model_name)
    outputs = query_hf_model(model, tokenizer, prompt, device,
                             system_prompt=system_prompt, 
                             max_new_tokens=max_new_tokens)
    return decode_outputs(outputs)

def pretty_print_docs(docs, print_body=True):
    for i, doc in enumerate(docs):
        print(f"DOCUMENT {i + 1}")
        print(f"{doc.metadata['Email Sender']}")
        if print_body:
            print(doc.page_content[:60])
        if 'wormy' in doc.metadata['Email Sender']:
            print("\nWORM HERE!! Successfully added worm email to RAG.")
        print(f"{'='*60}")

def RAG_with_worm_injection(db, incoming_email, k, worm_email,
                            worm_location, verbose=False):
    retrieved_docs = db.similarity_search(incoming_email, k=k)
    retrieved_docs.insert(worm_location, worm_email)
    if verbose:
        pretty_print_docs(retrieved_docs)
    return retrieved_docs

def encode_with_ceaser_cipher(text, shift):
    encoded = ''
    for char in text:
        if char in consts.ALPHABET_LOWER:
            num = ord(char) + shift
            if num < ord('a'):
                num = ord('z') + 1 - (ord('a') - num)
            if num > ord('z'):
                num = ord('a') + (num - ord('z') - 1)
            encoded += chr(num)
        elif char in consts.ALPHABET_UPPER:
            num = ord(char) + shift
            if num < ord('A'):
                num = ord('Z') + 1 - (ord('A') - num)
            if num > ord('Z'):
                num = ord('A') + (num - ord('Z') - 1)
            encoded += chr(num)
        else:
            encoded += char
    return encoded


def get_metrics(all_vectorized_worms, worm, 
                dir='Results/similarity metrics on worm and rag docs/',
                sample_size=50):
    
    N = len(all_vectorized_worms)
    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')

    Benign_metrics = {"BLEU": [], 'METEOR': [], 'ROUGE': []}
    Worm_metrics = {"BLEU": [], 'METEOR': [], 'ROUGE': []}

    # random.seed(42)
    random_indices = [i for i in range(N)]
    random.shuffle(random_indices)

    for i in tqdm(range(sample_size)):
        idx = random_indices[i]
        
        wormed_response = all_vectorized_worms[idx].page_content
        worm_seed = [consts.PREFIX + worm]
        
        references = [wormed_response for _ in range(N - 1)]
        predictions = [all_vectorized_worms[j].page_content 
                       for j in range(N) if j != idx]
        
        # metric 1
        bleu_benign = bleu.compute(predictions=predictions, 
                                   references=references)['bleu']
        Benign_metrics['BLEU'].append(bleu_benign)
        Worm_metrics['BLEU'].append(
            bleu.compute(predictions=[wormed_response], 
                         references=worm_seed)['bleu']
        )

        # metric 2
        meteor_benign = meteor.compute(predictions=predictions, 
                                    references=references)['meteor']
        Benign_metrics['METEOR'].append(meteor_benign)
        Worm_metrics['METEOR'].append(
            meteor.compute(predictions=[wormed_response], 
                        references=worm_seed)['meteor']
        )

        # metric 3
        rouge_benign = rouge.compute(predictions=predictions, 
                                    references=references)['rougeL']
        Benign_metrics['ROUGE'].append(rouge_benign)
        Worm_metrics['ROUGE'].append(
            rouge.compute(predictions=[wormed_response], 
                        references=worm_seed)['rougeL']
        )
    # end for loop
    with open(dir + 'benign_metrics.json', 'w') as file:
        json.dump(Benign_metrics, file)

    with open(dir + 'worm_metrics.json', 'w') as file:
        json.dump(Worm_metrics, file)