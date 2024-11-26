# PROJECT: PDF CHATBOT  

The **PDF Chatbot** is an AI-powered application that allows users to interact with PDF documents using natural language. It processes PDF files, extracts their content, and leverages a language model to provide context-aware responses to user queries about the document.  

---

## <strong>Features</strong>  

- **PDF Parsing**: Extracts text and metadata from uploaded PDF files.  
- **Intelligent Query Processing**: Uses an advanced language model to understand and respond to user queries.  
- **Interactive Web Interface**: Built with Gradio for seamless user interaction.  
- **Multilingual Support**: Handles documents and queries in multiple languages.  
- **Scalable**: Designed for single-file or batch processing.  

---

## <strong>Tech Stack</strong>  

### <strong>Languages and Frameworks</strong>  
- Python  

### <strong>Libraries</strong>  
- **PyPDF2**: For extracting text from PDF files.  
- **Gradio**: For creating the user-friendly interface.  
- **Hugging Face Transformers**: For language model processing.  
- **LangChain**: For building conversational AI workflows.  

---

## <strong>Project Workflow</strong>  

1. **File Upload**:  
   - Users upload a PDF file via the Gradio interface.  

2. **PDF Parsing**:  
   - The system extracts text and metadata using **PyPDF2**.  

3. **Query Processing**:  
   - The user inputs questions about the document.  
   - The chatbot processes the query using **LangChain** and a language model (e.g., GPT-based).  

4. **Response Generation**:  
   - The chatbot generates accurate and context-aware responses to the user’s questions.  

5. **Output Display**:  
   - Responses are displayed in the Gradio interface.  

---

## <strong>Setup and Installation</strong>  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/your-username/pdf-chatbot.git  
   cd pdf-chatbot  
