# RecruitMatch-AI
RecruitMatch AI is a GenAI-based application that automates resume screening for recruitment teams by intelligently ranking candidates against job descriptions. The system processes multiple resumes in PDF, DOCX, or TXT formats, extracting structured information through advanced NLP pipelines using spaCy for named entity recognition.

📋 Table of Contents
1. Project Overview

2. Features

3. Tech Stack

4. Project Architecture

5. Installation & Setup

6. Usage Guide

7. Workflow Diagrams

8. Detailed Component Breakdown

9.  Scoring Algorithm


🎯 Project Overview
RecruitMatch AI is an end-to-end AI-powered Resume Ranking and Candidate Evaluation System designed to automate and enhance the recruitment process. The system intelligently matches candidates' resumes against job descriptions using advanced Natural Language Processing (NLP), Semantic Similarity, and Machine Learning techniques.

What Problem Does It Solve?
Manual Resume Screening: Automatically evaluates and ranks resumes instead of manual review

Bias Reduction: Data-driven, objective scoring reduces human bias in hiring

Time Efficiency: Process 100+ resumes in seconds vs. hours of manual screening

Skill Matching: Intelligently identifies skill overlaps between candidates and requirements

Holistic Evaluation: Considers skills, experience, education, and overall fit simultaneously

Key Use Cases
✅ HR teams screening bulk applications
✅ Recruitment agencies ranking candidates
✅ Technical hiring panels evaluating candidates
✅ Career portals ranking job matches
✅ Talent acquisition optimization

✨ Features
Core Functionality
Feature	Description
Multi-Format Document Parsing	Supports PDF, DOCX, and TXT resume formats with robust error handling
Information Extraction	Automatically extracts name, contact info, skills, experience, education
Skill Recognition	200+ predefined skills across 8+ categories (Programming, Cloud, Data, etc.)
Semantic Matching	Uses advanced embeddings for contextual job-resume matching
Multi-Factor Scoring	Weighs 5 evaluation factors: Skills (40%), Experience (25%), Education (15%), Years (12%), LLM (8%)
LLM Integration	Optional HuggingFace API integration for additional contextual evaluation
Web UI	Interactive Gradio interface for easy resume upload and JD input
Ranked Results	HTML-formatted candidate rankings with score breakdowns
Custom Skills	Support for user-defined priority skills
Advanced Features
Named Entity Recognition (NER): Extracts names and entities using spaCy

Education Scoring Boost: Recognizes and rewards advanced degrees (Master's, PhD)

Years of Experience Normalization: Handles varying experience levels fairly

Batch Similarity Processing: Efficiently compares multiple resumes simultaneously

JSON Export: Detailed scoring data for downstream analysis

Error Handling: Graceful fallbacks if models/APIs unavailable

🛠️ Tech Stack
Core Machine Learning & NLP
Category	Technologies	Version/Details
Embeddings & Transformers	Sentence Transformers (all-MiniLM-L6-v2)	384-dim, lightweight (~90MB)
NLP Framework	spaCy	en_core_web_sm (3.8.0), for NER & tokenization
Text Processing	NLTK	Punkt, Averaged Perceptron Tagger, Stopwords, WordNet
Semantic Similarity	scikit-learn	TfidfVectorizer, Cosine Similarity
Numerical Computing	NumPy	Matrix operations, batch processing
Data Processing	Pandas	Data structuring (optional for larger-scale deployments)
Document Processing
Tool	Purpose
pdfplumber	Extract text from PDF documents page-by-page
python-docx	Parse Microsoft Word (.docx) files
re (Regex)	Text normalization and pattern matching
io (BytesIO)	In-memory file handling without disk I/O
Web Framework & API
Component	Technology	Details
Frontend/UI	Gradio	Interactive web interface, no HTML/CSS required
LLM Integration	HuggingFace Inference API	Google Flan-T5 Base model (optional)
HTTP Requests	requests	API communication with HuggingFace
Development & Deployment
Tool	Purpose
Google Colab	Development & execution environment
Jupyter Notebook	Interactive development with embedded progress bars
Python 3	Primary language (3.8+)
Data Structures & Utilities
Collections.Counter: Skill frequency analysis

datetime: Timestamp generation for results

json: Structured data export

tempfile: Temporary file handling for document processing

html: HTML escaping for safe output rendering

📐 Project Architecture
High-Level System Design
text
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT LAYER                         │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐    │
│  │   Resumes    │    │      JD      │   │ Custom Skills│    │
│  │(PDF/DOCX)    │    │   (Text)     │   │  (Optional)  │    │
│  └──────────────┘    └──────────────┘   └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│            DOCUMENT PARSING & EXTRACTION LAYER              │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   PDF Ext   │  │  DOCX Parse │  │   TXT Read  │          │
│  └────────┬────┘  └────────┬────┘  └────────┬────┘          │
│           │                │                │               │
│           └────────────────┼────────────────┘               │
│                            ↓                                │
│              ┌────────────────────────────┐                 │
│              │  Text Normalization        │                 │
│              │  (Regex Cleaning)          │                 │
│              └────────────┬───────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│        INFORMATION EXTRACTION & PARSING LAYER               │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │  NER +   │ │ Contact  │ │ Education│ │Experience│        │
│  │ Name     │ │ Extraction │ Parsing  │ │Extraction│        │
│  │Extraction│ └──────────┘ └──────────┘ └──────────┘        │
│  └──────────┘                                               │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │   SKILLS EXTRACTION (Multi-Strategy)│                    │
│  │  1. Exact Substring Matching        │                    │
│  │  2. Token-Based Matching            │                    │
│  │  3. NLP-Based Fallback              │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│      EMBEDDING & SEMANTIC SIMILARITY LAYER                  │
│                                                             │
│  ┌──────────────────────────────────────┐                   │
│  │ Sentence Transformers                │                   │
│  │ (all-MiniLM-L6-v2, 384-dim)         │                    │
│  │ - Resume embeddings                  │                   │
│  │ - JD embeddings                      │                   │
│  │ - Normalized L2 vectors              │                   │
│  └──────────────────────────────────────┘                   │
│                      ↓                                      │
│  ┌──────────────────────────────────────┐                   │
│  │ Cosine Similarity Computation        │                   │
│  │ (Batch processing for efficiency)    │                   │
│  └──────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           MULTI-FACTOR SCORING LAYER                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FACTOR 1: Skills Match (40%)                        │   │
│  │  Precision = (JD Skills Found in Resume) / (Total)   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FACTOR 2: Experience Relevance (25%)                │   │
│  │  Semantic similarity: JD vs. Resume Experience       │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FACTOR 3: Education Relevance (15%)                 │   │
│  │  Semantic match + bonus for advanced degrees         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FACTOR 4: Years of Experience (12%)                 │   │
│  │  Normalized: min(1.0, yearsExp / 10.0)               │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  FACTOR 5: LLM Overall Fit                           │   │
│  │  HuggingFace Inference API                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                      ↓                                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  WEIGHTED AGGREGATION                                │   │
│  │  Score = (S×0.40)+(E×0.25)+(Ed×0.15)+(Y×0.12)+(L×0.08)   │
│  │  Final Score (0-100) = Score × 100                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              RANKING & OUTPUT LAYER                         │
│                                                             │
│  ┌─────────────────────────────────────┐                    │
│  │ Sort candidates by final score      │                    │
│  │ Select top-K results                │                    │
│  └─────────────────────────────────────┘                    │
│                      ↓                                      │
│  ┌─────────────────────────────────────┐                    │
│  │ Format results with:                │                    │
│  │ - Rank, Name, Contact               │                    │
│  │ - Score breakdown                   │                    │
│  │ - Matched skills                    │                    │
│  │ - Education & Experience            │                    │
│  └─────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  GRADIO WEB INTERFACE                       │
│         (Results displayed to user in ranked order)         │
└─────────────────────────────────────────────────────────────┘
🚀 Installation & Setup
Prerequisites
Python 3.8+

Internet connection (for HuggingFace model downloads)

Google Colab (recommended) or local machine with 4GB+ RAM

Step 1: Install Dependencies
bash
# Install required packages
pip install -q gradio pdfplumber python-docx sentence-transformers faiss-cpu scikit-learn spacy nltk transformers requests

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download NLTK datasets
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('stopwords'); nltk.download('wordnet')"
Step 2: Configuration
python
# Set your HuggingFace token (optional, for LLM features)
HF_TOKEN = "hf_xxxxxxxxxxxxx"  # Get from huggingface.co/settings/tokens
HF_MODEL = "google/flan-t5-base"  # Can be changed to other HF models

# Models used
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim lightweight embeddings
NLP_MODEL = "en_core_web_sm"  # spaCy model for NER
Step 3: Run the Application
python
# In Google Colab or local Jupyter:
# The notebook automatically launches Gradio UI
# Open the public link to access the interface
💻 Usage Guide
Via Web Interface (Gradio)
Upload Resumes

Click "Upload Resumes" button

Select one or multiple resume files (PDF, DOCX, TXT)

Supports batch processing

Enter Job Description

Paste job description in text area

Minimum 50 characters required

Include key skills, responsibilities, requirements

Optional: Set Priority Skills

Enter comma-separated list of must-have skills

Example: Python, Machine Learning, AWS, Docker

Adjust Settings

Top-K: Select how many candidates to display (1-20)

Enable LLM Evaluation: Toggle for HuggingFace API scoring (requires valid token)

Click "Rank Resumes"

System processes resumes

Returns ranked list with scores and breakdowns
