# 🤖 IRA - Your Intelligent Document Q&A Companion

## 📖 Project Overview

IRA (Intelligent Reading Assistant) is an AI-powered document analysis and question-answering application built with Streamlit and Google's Gemini AI. Users can upload PDF documents, register accounts, and have intelligent conversations about their uploaded content using advanced RAG (Retrieval-Augmented Generation) technology.

## ✨ Features

### 🔐 User Management
- **Secure Registration & Login**: User authentication with hashed passwords
- **Personal Document Library**: Each user has their own private document collection
- **Session Management**: Persistent chat history and document associations

### 📄 Document Processing
- **PDF Upload & Storage**: Support for text-based PDF documents
- **Smart Text Extraction**: Automatic text extraction and chunking
- **Vector Embeddings**: Local sentence-transformer embeddings for fast similarity search
- **Document Management**: View, organize, and delete uploaded documents

### 🤖 AI-Powered Chat
- **General Chat**: Direct conversations with IRA using Gemini AI
- **Document Q&A**: Context-aware questions about uploaded documents
- **RAG Implementation**: Retrieval-Augmented Generation for accurate, source-based responses
- **Chat History**: Persistent conversation history per user and document

### 🎨 User Interface
- **Modern Design**: Clean, responsive Streamlit interface
- **Intuitive Navigation**: Easy-to-use sidebar menu system
- **Real-time Processing**: Live feedback during document processing
- **Mobile Friendly**: Responsive design that works on all devices

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - Google Gemini 2.0 Flash (Text Generation)
  - SentenceTransformers (Local Embeddings)
  - FAISS (Vector Search)
  - LangChain (RAG Pipeline)
- **Backend**: SQLite Database
- **Document Processing**: PyPDF2
- **Authentication**: Custom password hashing

## 📋 Prerequisites

- Python 3.8+
- Google API Key (for Gemini AI)
- 2GB+ RAM (for embedding models)

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Uday2kranth/IRA---Your-Intelligent-Document-Q-A-Companion-.git
cd IRA---Your-Intelligent-Document-Q-A-Companion-
```

### 2. Create Virtual Environment
```bash
python -m venv IRA_env
# Windows
IRA_env\Scripts\activate
# Mac/Linux
source IRA_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

**Get your Google API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

### 5. Run the Application
```bash
streamlit run main2.py
```

The app will open at `http://localhost:8501`

## 📱 Usage Guide

### Getting Started
1. **Register**: Create your account on the Login/Signup page
2. **Upload Documents**: Go to "Upload Documents" and add your PDF files
3. **Chat**: Start asking questions about your documents!

### Chat Features
- **General Chat**: Ask IRA anything on the "Chat With IRA" page
- **Document Chat**: Upload documents and ask specific questions about their content
- **Context Awareness**: IRA remembers your conversation history

### Document Management
- **View Content**: Preview your uploaded documents
- **Organize Files**: Keep track of all your uploaded materials
- **Delete Files**: Remove documents you no longer need

## 🏗️ Architecture

```
IRA Application
├── User Interface (Streamlit)
├── Authentication System (SQLite)
├── Document Processing (PyPDF2)
├── Vector Database (FAISS)
├── Embedding Generation (SentenceTransformers)
├── AI Response (Google Gemini)
└── Chat Management (Session State)
```

## 📊 Database Schema

### Users Table
- `user_id` (Primary Key)
- `first_name`, `last_name`
- `email` (Unique)
- `password` (Hashed)
- `date_of_birth`

### Files Table
- `file_id` (Primary Key)
- `user_id` (Foreign Key)
- `file_name`
- `file_path`

## 🔒 Security Features

- **Password Hashing**: SHA-256 encryption for user passwords
- **API Key Protection**: Environment variable storage for sensitive keys
- **User Isolation**: Each user can only access their own documents
- **Input Validation**: Comprehensive error handling and input sanitization

## 🎯 Use Cases

- **Academic Research**: Analyze research papers and ask detailed questions
- **Legal Documents**: Review contracts and legal documents with AI assistance
- **Technical Documentation**: Get explanations of complex technical content
- **Study Materials**: Interactive learning with textbooks and course materials
- **Business Reports**: Extract insights from business documents and reports

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed with `pip install -r requirements.txt`
2. **API Key Issues**: Verify your Google API key is correctly set in `.env`
3. **Memory Issues**: Restart the app if embedding models consume too much RAM
4. **PDF Processing**: Ensure PDFs contain extractable text (not scanned images)

### Debug Mode
Add this to your `.streamlit/config.toml`:
```toml
[logger]
level = "debug"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google AI**: Gemini AI API for intelligent responses
- **Hugging Face**: SentenceTransformers for embedding generation
- **Streamlit**: Amazing framework for rapid web app development
- **LangChain**: RAG implementation and document processing tools
- **FAISS**: Efficient vector similarity search

## 📞 Support

- **GitHub Issues**: [Create an issue](https://github.com/Uday2kranth/IRA---Your-Intelligent-Document-Q-A-Companion-/issues)
- **Email**: Contact the repository owner
- **Documentation**: Check this README and code comments

## 🚀 Future Enhancements

- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Cloud storage integration (AWS S3, Google Drive)
- [ ] Advanced user management and admin panel
- [ ] Real-time collaboration features
- [ ] Mobile app development
- [ ] Multi-language support
- [ ] Advanced analytics and usage tracking

---

**Made with ❤️ by [Uday2kranth](https://github.com/Uday2kranth)**

*Last updated: September 2025*