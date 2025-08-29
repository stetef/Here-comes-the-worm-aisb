# %%
import os
from dotenv import load_dotenv
import json

import matplotlib.pyplot as plt
import numpy as np

import consts
import utils

from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# %%
load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                   openai_api_key=os.getenv("OPENAI_API_KEY"))

# collect db with normal benign emails
db_no_worm = FAISS.load_local('Data/embedded_data/without_worm/', embedding_model, 
                              allow_dangerous_deserialization=True)
all_vectorized_docs = db_no_worm.similarity_search("random text", k=1000)  
print(len(all_vectorized_docs))

# collect db with worm prepended to all emails
db_worm = FAISS.load_local('Data/embedded_data/with_worm/', embedding_model, 
                            allow_dangerous_deserialization=True)
all_vectorized_worms = db_worm.similarity_search("random text", k=1000)
print(len(all_vectorized_worms))

# collect db with just the worm seeded email
vector_worm_db = FAISS.load_local('Data/embedded_data/vectorized_WORM/', embedding_model, 
                                  allow_dangerous_deserialization=True)
vectorized_worm = vector_worm_db.similarity_search("random text", k=10)
print(len(vectorized_worm))

# %%

# utils.get_metrics(all_vectorized_worms, consts.WORM, sample_size=50)

# %%

dir = 'Results/imilarity metrics on worm and rag docs/'

with open(dir + 'benign_metrics.json', 'r') as file:
    Benign_metrics = json.load(file)

with open(dir + 'worm_metrics.json', 'r') as file:
    Worm_metrics = json.load(file)

# %%

fig, axes = plt.subplots(figsize=(10, 4), ncols=3)
for i, key in enumerate(['BLEU', 'METEOR', 'ROUGE']):
    axes[i].grid(axis='both', zorder=0)
    bin_size = 0.05
    bins = np.arange(0, 1 + bin_size, bin_size)
    axes[i].hist(Benign_metrics[key], bins=bins, zorder=5, color='blue', 
                 lw=1, ec='w', alpha=0.5, label='Benign')
    axes[i].hist(Worm_metrics[key], bins=bins, zorder=5, color='red', 
                 lw=1, ec='w', alpha=0.5, label='Worm')
    leg = axes[i].legend(fontsize=16, bbox_to_anchor=(0.5, 0.97),
                        loc='lower center', ncol=2,
                        handletextpad=0.3, framealpha=0,
                        columnspacing=0.5, handlelength=1.0,
                        handleheight=0.7)
    xticks = [0, 0.25, 0.5, 0.75, 1.]
    xtick_labs = ['', 0.25, 0.5, 0.75, '']
    axes[i].set_xticks(xticks)
    axes[i].set_xticklabels(xtick_labs, fontsize=14)
    axes[i].tick_params(width=1.5, size=7, labelsize=14)
    axes[i].set_xlabel(key, fontsize=16)
    axes[i].set_xlim([0, 1])
axes[0].set_ylabel('Counts', fontsize=16)

plt.tight_layout()
plt.savefig('Figures/metrics_on_worm.png', dpi=400, bbox_inches='tight')

# %%

