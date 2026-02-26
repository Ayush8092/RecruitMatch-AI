# ============================================================
# utils.py
# Shared constants, skill database, education keywords,
# and the ParseQualityAuditor used across the app.
# ============================================================

import re
import warnings
from collections import Counter
from typing import Dict, List

warnings.filterwarnings('ignore')

# ── Availability flags (set at import time) ──────────────────
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── Skill Database ────────────────────────────────────────────
SKILL_DATABASE = [
    # Programming Languages
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'r', 'go',
    'rust', 'swift', 'kotlin', 'scala', 'perl', 'ruby', 'php', 'matlab', 'bash',
    # Web Technologies
    'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'express',
    'django', 'flask', 'fastapi', 'spring', 'springboot', 'laravel', 'next.js',
    'graphql', 'rest api', 'restful', 'jquery', 'bootstrap', 'tailwind',
    # Databases
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle',
    'sqlite', 'dynamodb', 'elasticsearch', 'neo4j', 'firebase', 'supabase',
    # Cloud Platforms
    'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'cloudflare',
    # DevOps & MLOps
    'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ci/cd',
    'terraform', 'ansible', 'linux', 'nginx', 'apache', 'mlflow', 'airflow',
    # AI / ML / DL / NLP
    'machine learning', 'deep learning', 'neural network', 'nlp',
    'natural language processing', 'computer vision', 'reinforcement learning',
    'transfer learning', 'generative ai', 'llm', 'large language model',
    'gpt', 'bert', 'transformers', 'diffusion models', 'rag',
    # Data Science
    'data science', 'data analysis', 'data engineering', 'data visualization',
    'statistics', 'feature engineering', 'etl', 'big data', 'spark', 'hadoop',
    'tableau', 'power bi', 'excel', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    # ML Frameworks & Libraries
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'sklearn', 'xgboost',
    'lightgbm', 'catboost', 'huggingface', 'openai', 'langchain', 'llamaindex',
    'opencv', 'spacy', 'nltk', 'gensim', 'fastai',
    # Software Engineering
    'oop', 'design patterns', 'microservices', 'agile', 'scrum', 'jira',
    'unit testing', 'tdd', 'system design', 'api design', 'kafka', 'rabbitmq',
]

# ── Education Keywords ────────────────────────────────────────
EDUCATION_KEYWORDS = [
    'b.tech', 'btech', 'm.tech', 'mtech', 'b.e', 'be', 'm.e', 'me',
    'b.sc', 'bsc', 'm.sc', 'msc', 'phd', 'ph.d', 'mba', 'bca', 'mca',
    'bachelor', 'master', 'doctorate', 'diploma', 'pgdm', 'b.com', 'bcom',
    'b.a', 'ba', 'm.a', 'ma', 'engineering', 'science', 'technology',
    'computer science', 'information technology', 'electronics',
    'mechanical', 'electrical', 'civil', 'chemical', 'mathematics',
    'statistics', 'data science', 'artificial intelligence', 'machine learning',
    'iit', 'nit', 'bits', 'vtu', 'university', 'institute', 'college',
]


# ── ParseQualityAuditor ───────────────────────────────────────
class ParseQualityAuditor:
    """
    Audits parsed resume dicts and assigns a parse confidence score (0-100).

    Confidence scoring:
      +25   Name detected
      +20   Email detected
      +15   Phone detected
      +20   Skills detected (>=5 full, >=2 half)
      +10   Experience > 0
      +10   Education detected
    """

    WEIGHTS = {
        'name':       25,
        'email':      20,
        'phone':      15,
        'skills':     20,
        'experience': 10,
        'education':  10,
    }

    LOW_CONFIDENCE_THRESHOLD = 50
    BATCH_FAIL_WARN_PCT      = 40

    def __init__(self):
        self.audit_log: List[Dict] = []

    def audit(self, parsed: Dict) -> Dict:
        scores  = {}
        reasons = []

        name = parsed.get('name', '')
        if name and name not in ('Name Not Found', 'Unknown', ''):
            scores['name'] = self.WEIGHTS['name']
        else:
            scores['name'] = 0
            reasons.append('Name not detected')

        email = parsed.get('email', 'Not Found')
        if email and email != 'Not Found' and '@' in email:
            scores['email'] = self.WEIGHTS['email']
        else:
            scores['email'] = 0
            reasons.append('Email not detected')

        phone = parsed.get('phone', 'Not Found')
        if phone and phone != 'Not Found':
            scores['phone'] = self.WEIGHTS['phone']
        else:
            scores['phone'] = 0
            reasons.append('Phone not detected')

        skill_count = len(parsed.get('skills', []))
        if skill_count >= 5:
            scores['skills'] = self.WEIGHTS['skills']
        elif skill_count >= 2:
            scores['skills'] = int(self.WEIGHTS['skills'] * 0.5)
            reasons.append(f'Only {skill_count} skill(s) detected (low)')
        else:
            scores['skills'] = 0
            reasons.append('No skills detected — likely parse failure')

        exp_months = parsed.get('experience_years', 0) * 12
        if exp_months > 0:
            scores['experience'] = self.WEIGHTS['experience']
        else:
            scores['experience'] = 0
            reasons.append('Experience not detected')

        edu = parsed.get('education_display', 'Not Found')
        if edu and edu != 'Not Found':
            scores['education'] = self.WEIGHTS['education']
        else:
            scores['education'] = 0
            reasons.append('Education not detected')

        total        = sum(scores.values())
        is_low_conf  = total < self.LOW_CONFIDENCE_THRESHOLD
        parse_engine = parsed.get('parse_engine', 'unknown')

        audit_result = {
            'file':           parsed.get('file', 'unknown'),
            'confidence':     total,
            'low_confidence': is_low_conf,
            'field_scores':   scores,
            'issues':         reasons,
            'parse_engine':   parse_engine,
            'skill_count':    skill_count,
        }
        self.audit_log.append(audit_result)
        return audit_result

    def batch_report(self) -> Dict:
        if not self.audit_log:
            return {'message': 'No parse audits recorded yet.'}

        total      = len(self.audit_log)
        low_conf   = [a for a in self.audit_log if a['low_confidence']]
        low_pct    = len(low_conf) / total * 100
        avg_conf   = sum(a['confidence'] for a in self.audit_log) / total
        engine_cts = Counter(a['parse_engine'] for a in self.audit_log)

        return {
            'total_parsed':         total,
            'average_confidence':   round(avg_conf, 1),
            'low_confidence_count': len(low_conf),
            'low_confidence_pct':   round(low_pct, 1),
            'batch_warning':        low_pct >= self.BATCH_FAIL_WARN_PCT,
            'engines_used':         dict(engine_cts),
            'low_confidence_files': [
                {
                    'file':       a['file'],
                    'confidence': a['confidence'],
                    'issues':     a['issues'],
                }
                for a in low_conf
            ],
        }

    def get_field_failure_rates(self) -> Dict:
        if not self.audit_log:
            return {}
        total = len(self.audit_log)
        return {
            field: round(
                sum(1 for a in self.audit_log
                    if a['field_scores'].get(field, 0) == 0)
                / total * 100, 1)
            for field in self.WEIGHTS
        }

    def reset(self):
        self.audit_log = []


# ── Singleton auditor shared across requests ─────────────────
auditor = ParseQualityAuditor()