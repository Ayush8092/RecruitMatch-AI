RecruitMatch AI - Resume Ranking System
Overview

RecruitMatch AI is a comprehensive resume ranking and matching system designed to streamline recruitment by intelligently analyzing candidate resumes against job descriptions. The system combines OCR capabilities for PDF extraction, semantic text analysis using sentence transformers, and cosine similarity matching to rank candidates based on their fit for specific roles.

This notebook documents the complete development pipeline from data collection and preprocessing to model training, evaluation, and deployment.

The fine-tuned model powers a live Gradio application deployed at:
**https://huggingface.co/spaces/Ak47-model-ml/RecruitMatch_AI**


Key Features

i. PDF Resume Parsing: Extracts text from PDF resumes using pdfplumber, pdf2image, and pytesseract OCR

ii. Semantic Embeddings: Converts resumes and job descriptions into dense vector representations using sentence-transformers

iii. Intelligent Matching: Ranks candidates using cosine similarity between resume embeddings and job requirement embeddings

iv. Production Ready: Deployed as a Gradio web application with full Docker support for Hugging Face Spaces







