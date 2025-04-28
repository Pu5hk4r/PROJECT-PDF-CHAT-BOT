# PROJECT: PDF CHATBOT

The **PDF Chatbot** is an AI-powered application that allows users to interact with PDF documents using natural language. It processes PDF files, extracts their content, and leverages a language model to provide context-aware responses to user queries about the document.

---

## SCREENSHOTS
![UserInterface](https://github.com/Pu5hk4r/PROJECT-PDF-CHAT-BOT/blob/main/pdf_chatbot.png)
![](https://github.com/Pu5hk4r/PROJECT-PDF-CHAT-BOT/blob/main/pdf_bot.png)
![Result](https://github.com/Pu5hk4r/PROJECT-PDF-CHAT-BOT/blob/main/pdfbot.png)

## **Features**

- **PDF Parsing**: Extracts text and metadata from uploaded PDF files.
- **Intelligent Query Processing**: Uses an advanced language model to understand and respond to user queries.
- **Interactive Web Interface**: Built with Gradio for seamless user interaction.
- **Multilingual Support**: Handles documents and queries in multiple languages.
- **Scalable**: Designed for single-file or batch processing.

---

## **Tech Stack**

### **Languages and Frameworks**
- Python
- LLM Models Mistral and OLlama
- Natural Language Processing

### **Libraries**
- **PyPDF2**: For extracting text from PDF files.
- **Gradio**: For creating the user-friendly interface.
- **Hugging Face Transformers**: For language model processing.
- **LangChain**: For building conversational AI workflows.
- **TensorFlow**: For building Deep Learning Neural Network Models
- **Keras**: Deep Learning API

---

## **Project Workflow**

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

## **Setup and Installation**

1. Clone the repository:
   ```bash
 https://github.com/Pu5hk4r/PROJECT-PDF-CHAT-BOT.git
   cd pdf-chatbot
