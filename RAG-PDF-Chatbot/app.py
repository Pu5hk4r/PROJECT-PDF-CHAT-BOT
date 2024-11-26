import gradio as gr
import os
api_token = os.getenv("HF_TOKEN")


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
import torch

list_llm = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]  
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load and split PDF document
def load_doc(list_file_path):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024, 
        chunk_overlap = 64 
    )  
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Create vector database
def create_db(splits):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    if llm_model == "meta-llama/Meta-Llama-3-8B-Instruct":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            huggingfacehub_api_token = api_token,
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    else:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token = api_token,
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )

    retriever=vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

# Initialize database
def initialize_database(list_file_obj, progress=gr.Progress()):
    # Create a list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # Load document and create splits
    doc_splits = load_doc(list_file_path)
    # Create or load vector database
    vector_db = create_db(doc_splits)
    return vector_db, "Database created!"

# Initialize LLM
def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "QA chain initialized. Chatbot is ready!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    # Generate response using QA chain
    response = qa_chain.invoke({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page
    

def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    return list_file_path


def demo():
    # with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as demo:
    with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue = "sky")) as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        gr.HTML("<center><h1>RAG PDF chatbot</h1><center>")
        gr.Markdown("""<b>Query your PDF documents!</b> This AI agent is designed to perform retrieval augmented generation (RAG) on PDF documents. The app is hosted on Hugging Face Hub for the sole purpose of demonstration. \
        <b>Please do not upload confidential documents.</b>
        """)
        with gr.Row():
            with gr.Column(scale = 86):
                gr.Markdown("<b>Step 1 - Upload PDF documents and Initialize RAG pipeline</b>")
                with gr.Row():
                    document = gr.Files(height=300, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF documents")
                with gr.Row():
                    db_btn = gr.Button("Create vector database")
                with gr.Row():
                        db_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Vector database status", 
                gr.Markdown("<style>body { font-size: 16px; }</style><b>Select Large Language Model (LLM) and input parameters</b>")
                with gr.Row():
                    llm_btn = gr.Radio(list_llm_simple, label="Available LLMs", value = list_llm_simple[0], type="index") # info="Select LLM", show_label=False
                with gr.Row():
                    with gr.Accordion("LLM input parameters", open=False):
                        with gr.Row():
                            slider_temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.5, step=0.1, label="Temperature", info="Controls randomness in token generation", interactive=True)
                        with gr.Row():
                            slider_maxtokens = gr.Slider(minimum = 128, maximum = 9192, value=4096, step=128, label="Max New Tokens", info="Maximum number of tokens to be generated",interactive=True)
                        with gr.Row():
                                slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k", info="Number of tokens to select the next token from", interactive=True)
                with gr.Row():
                    qachain_btn = gr.Button("Initialize Question Answering Chatbot")
                with gr.Row():
                        llm_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Chatbot status", 

            with gr.Column(scale = 200):
                gr.Markdown("<b>Step 2 - Chat with your Document</b>")
                chatbot = gr.Chatbot(height=505)
                with gr.Accordion("Relevent context from the source document", open=False):
                    with gr.Row():
                        doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                        source1_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                        source2_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                        source3_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a question", container=True)
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.ClearButton([msg, chatbot], value="Clear")
            
        # Preprocessing events
        db_btn.click(initialize_database, \
            inputs=[document], \
            outputs=[vector_db, db_progress])
        qachain_btn.click(initialize_LLM, \
            inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], \
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)

        # Chatbot events
        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        clear_btn.click(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()