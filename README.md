🚀 AI Study Coach Portfolio App

This project delivers a personalized learning experience through an AI coach application. It combines RAG (Retrieval-Augmented Generation) technology with an LSTM prediction model to support customized user study.

Key Features

RAG Knowledge Chatbot: Answers user questions based on uploaded documents (PDF, TXT, HTML).

Custom Content Generation: Generates summaries, quizzes, and practical example ideas tailored to specific topics and difficulty levels using the Gemini LLM.

LSTM Achievement Prediction: Provides motivation by predicting future learning achievement based on hypothetical historical study data.

AI Customer Support Simulator: Virtual scenarios for handling customer inquiries via chat, email, and phone with AI-generated response guidelines and hints.

Company Information & FAQ Management: Search and manage company-specific detailed information, popular products, trending news, and FAQs with multi-language support (Korean, English, Japanese).

Customer Inquiry Review: Allows agents to reconfirm customer inquiries with supervisors and generate AI answers and hints. Supports file attachments (images, PDFs) with automatic OCR and translation features.

Technical Stack

Frontend/App Framework: Streamlit

Backend/ML/LLM: Google Gemini API, LangChain

Vector Database: FAISS (Index stored in Firestore)

Persistent Storage: Google Cloud Firestore (Permanent data management via Admin SDK)

Deployment: Streamlit Cloud

🔑 Deployment Setup (Secrets)

This app requires the following two Secrets keys for deployment on Streamlit Cloud. (Sensitive information is not included in this repository.)

GEMINI_API_KEY: The personal key for making Gemini API calls.

FIREBASE_SERVICE_ACCOUNT_JSON: The complete Service Account JSON string for accessing Firestore via the Admin SDK.

Script for Japansese RAG version

<img width="1872" height="730" alt="image" src="https://github.com/user-attachments/assets/d4aaf6bc-168c-4ed9-8bb9-38dde0759a6c" />

Script for English RAG version

<img width="1890" height="830" alt="image" src="https://github.com/user-attachments/assets/eb94594a-ff80-4609-b2af-ad2a3f9c0a7d" />

Utilizing LSTM

<img width="1836" height="751" alt="image" src="https://github.com/user-attachments/assets/7f8c6ffd-a0f0-40da-994d-6f06859e2520" />

Asking children's case via Korean

<img width="1058" height="726" alt="image" src="https://github.com/user-attachments/assets/b05d168d-f98b-4ab9-a86e-78ce07a1595d" />


Movies for RAG and LSTM via my streamlit app

https://www.youtube.com/watch?v=MOQ2qD7Ws-E

## 🆕 Recent Updates (2025-12-08)

### Multi-language Support Improvements
- ✅ Fixed company information section to properly support Korean, English, and Japanese languages
- ✅ Added language-specific default FAQs and error messages
- ✅ Improved font handling for PDF downloads (Korean character support)

### File Upload & Translation Features
- ✅ Added file uploader for customer attachments (images, PDFs, screenshots)
- ✅ Automatic OCR for image files using Gemini Vision API
- ✅ Automatic translation for Korean files uploaded in Japanese/English versions
- ✅ Support for non-refundable travel product cancellation cases with evidence attachments

### UI/UX Improvements
- ✅ Improved copy functionality with larger font size (18px) for better readability
- ✅ Enhanced text display using HTML/CSS for better text selection and copying
- ✅ Added download buttons as alternative to copy functionality

### PDF Export Enhancements
- ✅ Fixed Korean character encoding issues in PDF downloads
- ✅ Improved font registration and application logic
- ✅ Enhanced font detection and fallback mechanisms
- ✅ Better error handling and debugging information

## 🔧 CI/CD & Development

This project includes automated CI/CD pipelines using GitHub Actions:

- **Python Package Testing**: Automated builds and tests across Python 3.9-3.12
- **Code Quality**: Pylint integration for code quality checks
- **Application Testing**: Syntax checks and import validation

See `GITHUB_PUSH_GUIDE.md` for detailed CI/CD setup instructions.

## 📋 Usage Guide

### Company Information & FAQ
1. Navigate to "Company Information & FAQ" tab
2. Search for a company name or select from the dropdown
3. View company information, popular products, trending news, and FAQs
4. Use the search function to find specific information

### Customer Inquiry Review
1. Navigate to "Customer Inquiry Review" tab
2. Select a company (optional)
3. Enter customer inquiry content
4. Upload attachments (images, PDFs, screenshots) if needed
   - For non-refundable travel products, attach evidence documents (flight delays, passport issues, etc.)
5. Generate AI answer or response hints
6. Copy or download the generated content

### File Upload & Translation
- Supported file types: PNG, JPG, JPEG, PDF, TXT
- Korean files uploaded in Japanese/English versions are automatically translated
- Image files are processed with OCR to extract text content
- Extracted text is included in AI answer and hint generation prompts








