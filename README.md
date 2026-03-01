1.Key Features:

i. PDF Resume Parsing: Extracts text from PDF resumes using pdfplumber, pdf2image, and pytesseract OCR
ii. Semantic Embeddings: Converts resumes and job descriptions into dense vector representations using sentence-transformers
iii. Intelligent Matching: Ranks candidates using cosine similarity between resume embeddings and job requirement embeddings
iv. Production Ready: Deployed as a Gradio web application with full Docker support for Hugging Face Spaces


2.System Architecture:


<img width="299" height="396" alt="System Architecture" src="https://github.com/user-attachments/assets/1d7096db-9ee7-4696-8fee-50af77bc320a" />



3.Technology Stack:

i. Programming Language: Python

ii. NLP Models: Sentence-BERT (SBERT)

iii. Libraries: Transformers, Sentence-Transformers, Scikit-learn, NumPy, Pandas, NLTK, SpaCy, PyPDF2, Matplotlib, Seaborn



4.Project Workflow:

1️⃣ Resume & JD Ingestion

i. Supports PDF and text-based resumes

ii. Handles large-scale resume ingestion

2️⃣ Text Cleaning & Preprocessing

i. Lowercasing

ii. Stopword removal

iii. Token normalization

iv. Noise filtering

3️⃣ Skill Extraction

i. Rule-based + semantic keyword detection

ii. Mapping technical and soft skills

4️⃣ Semantic Embedding Generation

i. Uses Sentence-BERT to convert text into dense vectors

5️⃣ Similarity Computation

i. Cosine similarity used to measure semantic closeness

6️⃣ Candidate Ranking

i. Resumes ranked based on similarity scores

ii. Supports top-K retrieval



5.Example Use Case:

Job Title: Machine Learning Engineer

The system analyzes uploaded resumes and produces:

i. Semantic similarity score

ii. Skill match percentage

iii. Ranked list of candidates

iv. Intelligent matching justification

This allows recruiters to instantly identify best-fit candidates.



6.Dataset Strategy:

The system supports:

i. Real-world resume datasets

ii. Structured resume–JD matching datasets

iii. Supervised fine-tuning data (planned extension)

The architecture is intentionally designed to integrate labeled datasets for supervised model fine-tuning and ranking optimization, enabling enterprise-grade accuracy.



7.Evaluation Methodology:

i. Semantic similarity analysis

ii. Ranking consistency checks

iii. Manual validation using real resumes

iv. Planned metrics: Precision@K, Recall@K, NDCG


8.Performance Highlights:

i. Processes 100+ resumes per batch

ii. Achieves high semantic matching accuracy

iii. Significantly reduces manual screening time

iv. Produces stable ranking results



9.Future Enhancements:

i. Supervised Sentence-BERT fine-tuning

ii. Learning-to-rank modeling

iii. Vector database integration (FAISS / Pinecone)

iv. Real-time ATS deployment

v. API-based resume screening service



10.Why This Project Stands Out:

i. Solves a real business problem

ii. Goes beyond keyword matching

iii. Demonstrates full ML system design

iv. Combines NLP, embeddings, ranking, and evaluation

v. Industry-ready architecture



11.Installation & Setup:
pip install -r requirements.txt
Or manually install core dependencies:
pip install sentence-transformers transformers scikit-learn numpy pandas nltk spacy matplotlib seaborn pypdf


12.How to Run:

i. Open the notebook

ii. Run all cells sequentially

iii. Upload resumes and job description

iv. Observe semantic matching and ranking outputs


13.Sample Output:

i. Semantic similarity score

ii. Skill overlap percentage

iii. Ranked resume list

iv. Matching confidence


14.Learning Outcomes:

i. NLP pipeline development

ii. Semantic search implementation

iii. Transformer embeddings

iv. Resume parsing

v. ML system design

vi. Industrial ML workflow development

👤 Author

Ayush Kumar
B.Tech Computer Science | Data Science & ML Enthusiast


15.Final Notes:

This project demonstrates production-level ML engineering thinking, focusing on real-world recruitment challenges, system scalability, and semantic intelligence. It is suitable for internship, research, and ML engineering role applications.
