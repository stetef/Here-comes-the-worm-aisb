import os
from dotenv import load_dotenv

import anthropic
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import consts

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