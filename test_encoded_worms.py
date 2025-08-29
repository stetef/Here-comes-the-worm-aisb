import consts
import utils

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# %%

# test ceasar cipher
k = 4
worm_loc = 0

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                   openai_api_key=os.getenv("OPENAI_API_KEY"))
templated_query = ChatPromptTemplate.from_template(consts.INCOMING_EMAIL_TEMPLATE)
db_no_worm = FAISS.load_local(consts.BENIGN_DB_DIR, embedding_model, 
                              allow_dangerous_deserialization=True)
worm_email = Document(page_content="Hiya!" + consts.WORM, id='imaworm123',
                      metadata={"Email Sender": consts.ATTACKER_EMAIL})

# generate incoming email with set sender and prefix
sender = 'test_employee@eenron.com'
body = 'Hey I need your help with something, can you help me?'
incoming_email = f'Email Sender: {sender} \n Email Body: {body}'

encoded_WORM = utils.encode_with_ceaser_cipher(consts.WORM, -7)
encoded_worm_email = Document(page_content=consts.PREFIX + encoded_WORM, 
                              id='imaworm123',
                              metadata={"Email Sender": 
                                        consts.ATTACKER_SENDER_EMAIL})

docs = utils.RAG_with_worm_injection(db_no_worm, incoming_email, k, 
                                     encoded_worm_email, worm_loc)
prompt = consts.TEMPLATED_QUERY.format(context=docs, incoming_email=incoming_email)
response = utils.call_open_ai(prompt, max_tokens=1024, model_name='gpt-5-mini-2025-08-07')

# %%
