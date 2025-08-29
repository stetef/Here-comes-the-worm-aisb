import os
import random
import re
from tqdm import tqdm

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

import consts

def collate_enron_files(data_dir):
    """Process Enron dataset into json-style dictionary."""
    def process_dir(dir, emails, key):
        for file in os.listdir(dir):
            if not os.path.isdir(dir + file):
                with open(dir + file, "r") as f:
                    try:
                        content = f.read()
                        sender_pattern = r'From:\s*([^\s,]+@[^\s,]+),?'
                        body_pattern = r'X-FileName:.*?\n(.*)'
                        sender_match = re.search(sender_pattern, content)
                        body_match = re.search(body_pattern, content, re.DOTALL)
                        if sender_match and body_match:
                            sender = sender_match.group(1)
                            body = body_match.group(1)
                            emails[key].append({"from": sender, 
                                                "body": body})
                        else:
                            pass
                    except Exception:
                        pass
                f.close()
            else:
                # not a directory
                pass
        # end for loop through every file in directory
    # end process_dir function
    enron_emails = {}
    directories = [employee_dir for employee_dir in os.listdir(data_dir)]
    for i in tqdm(range(len(directories))):
        employee_dir = directories[i]
        inbox_dir = f"{data_dir}{employee_dir}/inbox/"
        sent_dir = f"{data_dir}{employee_dir}/sent/"
        sent_items_dir = f"{data_dir}{employee_dir}/sent_items/"
        if os.path.isdir(inbox_dir) and os.path.isdir(sent_dir):
            emails = {"received": [],
                      "sent": []}
            process_dir(inbox_dir, emails, 'received')
            process_dir(sent_dir, emails, 'sent')
            process_dir(sent_items_dir, emails, 'sent')
            enron_emails[employee_dir] = emails
        else:
            pass
    return enron_emails

# Collect data into json-style dictionary
data = collate_enron_files("Data/maildir/")
print(f'Collected emails for a total of {len(data.keys())} employees...')

# %%

def embed_emails_from_data_dictionary(data, embedding_model, worm_email=None,
                                      out_dir='Data/embedded_data/') -> None:
    """Construct local database by storing embedded documents for each employee."""
    N = len(data)
    for i in tqdm(range(N)):
        employee = list(data.keys())[i]
        employee_database = []
        for email_category in ['received', 'sent']:
            for email in data[employee][email_category]:
                doc = Document(page_content=email['body'], 
                               metadata={"Email Sender": email['from']})
                employee_database.append(doc) 
            # end iteratre through emails
        # end go through each email category 
        if worm_email:
            # seed each eployee database with one worm email if passed in
            employee_database.append(worm_email)
        # shuffle entire dataset
        random.shuffle(employee_database)
        try:
            DB = FAISS.from_documents(employee_database, 
                                      embedding_model)  # vectorized database
            DB.save_local(out_dir)  # dump a pickle file to local directory 
        except Exception:
            pass
        if i >= 10:  # only collect for 10 employees for now
            break
    # end for loop over every employee

# Construct a vectorized database for each employee,
# where we will add one worm email to the inbox of each person
worm_email = Document(page_content=consts.ATTACK_PREFIX + consts.WORM, 
                      id='imaworm123',
                      metadata={"Email Sender": consts.ATTACKER_EMAIL})
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                   openai_api_key=os.getenv("OPENAI_API_KEY"))

embed_emails_from_data_dictionary(data, embedding_model, worm_email=worm_email, 
                                  out_dir=consts.WORM_DB_DIR)
embed_emails_from_data_dictionary(data, embedding_model, worm_email=None, 
                                  out_dir=consts.BENIGN_DB_DIR)

# make db with just vectorized worm
worm_email = Document(page_content="Hiya!" + consts.WORM, id='imaworm123',
                      metadata={"Email Sender": consts.ATTACKER_EMAIL})
DB = FAISS.from_documents([worm_email], embedding_model)
DB.save_local('Data/embedded_data/vectorized_WORM/')


# make actual db with worm prepended to real emails
def make_vectorized_worm_db(all_vectorized_docs, embedding_model,
                            out_dir='Data/embedded_data/with_worm/'):
    wormy_docs = []
    for doc in all_vectorized_docs:
        body = doc.page_content
        wormy_doc = Document(page_content=consts.WORM + body, 
                             metadata={"Email Sender": consts.ATTACKER_EMAIL})
        wormy_docs.append(wormy_doc)
    random.shuffle(wormy_docs)
    try:
        DB = FAISS.from_documents(wormy_docs, embedding_model)
        DB.save_local(out_dir)
    except Exception:
        pass

db_no_worm = FAISS.load_local('Data/embedded_data/without_worm/', embedding_model, 
                              allow_dangerous_deserialization=True)
all_vectorized_docs = db_no_worm.similarity_search("random text", k=1000)  
make_vectorized_worm_db(all_vectorized_docs, embedding_model)
