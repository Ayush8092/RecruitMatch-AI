# ============================================================
# model.py
# Contains all classes and singletons.
# Imported by app.py
# ============================================================

import os
import re
import json
import time
import math
import traceback
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing  import List, Dict, Optional, Tuple, Any
from collections import Counter

warnings.filterwarnings('ignore')

# ── Optional dependency flags ─────────────────────────────────

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path, convert_from_bytes
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


# ── Skill & Education databases ───────────────────────────────

SKILL_DATABASE = [
    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'r', 'go',
    'rust', 'swift', 'kotlin', 'scala', 'perl', 'ruby', 'php', 'matlab', 'bash',
    'html', 'css', 'react', 'angular', 'vue', 'node.js', 'nodejs', 'express',
    'django', 'flask', 'fastapi', 'spring', 'springboot', 'laravel', 'next.js',
    'graphql', 'rest api', 'restful', 'jquery', 'bootstrap', 'tailwind',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'oracle',
    'sqlite', 'dynamodb', 'elasticsearch', 'neo4j', 'firebase', 'supabase',
    'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean', 'cloudflare',
    'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'ci/cd',
    'terraform', 'ansible', 'linux', 'nginx', 'apache', 'mlflow', 'airflow',
    'machine learning', 'deep learning', 'neural network', 'nlp',
    'natural language processing', 'computer vision', 'reinforcement learning',
    'transfer learning', 'generative ai', 'llm', 'large language model',
    'gpt', 'bert', 'transformers', 'diffusion models', 'rag',
    'data science', 'data analysis', 'data engineering', 'data visualization',
    'statistics', 'feature engineering', 'etl', 'big data', 'spark', 'hadoop',
    'tableau', 'power bi', 'excel', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'sklearn', 'xgboost',
    'lightgbm', 'catboost', 'huggingface', 'openai', 'langchain', 'llamaindex',
    'opencv', 'spacy', 'nltk', 'gensim', 'fastai',
    'oop', 'design patterns', 'microservices', 'agile', 'scrum', 'jira',
    'unit testing', 'tdd', 'system design', 'api design', 'kafka', 'rabbitmq',
]

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
            reasons.append('⚠️  Name not detected')

        email = parsed.get('email', 'Not Found')
        if email and email != 'Not Found' and '@' in email:
            scores['email'] = self.WEIGHTS['email']
        else:
            scores['email'] = 0
            reasons.append('⚠️  Email not detected')

        phone = parsed.get('phone', 'Not Found')
        if phone and phone != 'Not Found':
            scores['phone'] = self.WEIGHTS['phone']
        else:
            scores['phone'] = 0
            reasons.append('⚠️  Phone not detected')

        skill_count = len(parsed.get('skills', []))
        if skill_count >= 5:
            scores['skills'] = self.WEIGHTS['skills']
        elif skill_count >= 2:
            scores['skills'] = int(self.WEIGHTS['skills'] * 0.5)
            reasons.append(f'⚠️  Only {skill_count} skill(s) detected (low)')
        else:
            scores['skills'] = 0
            reasons.append('❌ No skills detected — likely parse failure')

        exp_months = parsed.get('experience_years', 0) * 12
        if exp_months > 0:
            scores['experience'] = self.WEIGHTS['experience']
        else:
            scores['experience'] = 0
            reasons.append('⚠️  Experience not detected')

        edu = parsed.get('education_display', 'Not Found')
        if edu and edu != 'Not Found':
            scores['education'] = self.WEIGHTS['education']
        else:
            scores['education'] = 0
            reasons.append('⚠️  Education not detected')

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

    def batch_report(self) -> str:
        if not self.audit_log:
            return 'No parse audits recorded yet.'

        total      = len(self.audit_log)
        low_conf   = [a for a in self.audit_log if a['low_confidence']]
        low_pct    = len(low_conf) / total * 100
        avg_conf   = sum(a['confidence'] for a in self.audit_log) / total
        engine_cts = Counter(a['parse_engine'] for a in self.audit_log)

        lines = [
            '─' * 60,
            '📊 PARSE QUALITY AUDIT REPORT',
            '─' * 60,
            f'Total resumes parsed : {total}',
            f'Average confidence   : {avg_conf:.1f}/100',
            f'Low-confidence (<50) : {len(low_conf)} ({low_pct:.1f}%)',
            '',
            '🔧 Extraction engines used:',
        ]
        for engine, count in sorted(engine_cts.items(), key=lambda x: -x[1]):
            lines.append(f'  {engine:20s}: {count} files')

        if low_pct >= self.BATCH_FAIL_WARN_PCT:
            lines += [
                '',
                f'🚨 WARNING: {low_pct:.0f}% of resumes had low-confidence parses.',
                '   • Multi-column / table-heavy PDF layouts',
                '   • Scanned or image-based PDFs (OCR needed)',
                '   • Non-standard section headings',
            ]
        elif len(low_conf) > 0:
            lines += ['', f'⚠️  {len(low_conf)} resume(s) had low-confidence parses:']
            for a in low_conf:
                lines.append(f'  • {a["file"]} ({a["confidence"]}/100) '
                             f'— {"; ".join(a["issues"][:2])}')

        lines.append('─' * 60)
        return '\n'.join(lines)

    def get_field_failure_rates(self) -> Dict:
        if not self.audit_log:
            return {}
        total = len(self.audit_log)
        rates = {}
        for field in self.WEIGHTS:
            failed = sum(1 for a in self.audit_log
                        if a['field_scores'].get(field, 0) == 0)
            rates[field] = round(failed / total * 100, 1)
        return rates

    def reset(self):
        self.audit_log = []


# ── ResumeParser ──────────────────────────────────────────────

class ResumeParser:

    def __init__(self):
        self.skill_database     = SKILL_DATABASE
        self.education_keywords = EDUCATION_KEYWORDS
        self.auditor            = ParseQualityAuditor()
        print('[ResumeParser] Initialized.')
        print(f'  pdfplumber : {"✅" if PDFPLUMBER_AVAILABLE else "❌"}')
        print(f'  OCR        : {"✅" if OCR_AVAILABLE else "❌"}')

    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        if not PDFPLUMBER_AVAILABLE:
            return ''
        text = ''
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        tables = page.extract_tables()
                        table_text = ''
                        for table in tables:
                            for row in table:
                                row_cells = [str(c).strip() if c else '' for c in row]
                                table_text += ' | '.join(row_cells) + '\n'
                        page_text = page.extract_text(
                            x_tolerance=3, y_tolerance=3,
                            layout=True, x_density=7.25, y_density=13,
                        ) or ''
                        text += (table_text + '\n' + page_text).strip() + '\n'
                    except Exception as e:
                        print(f'  [pdfplumber] Page {page_num} error: {e}')
            return text.strip()
        except Exception as e:
            print(f'  [pdfplumber] File error: {e}')
            return ''

    def extract_text_pypdf2(self, pdf_path: str) -> str:
        if not PYPDF2_AVAILABLE:
            return ''
        text = ''
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + '\n'
                    except Exception as e:
                        print(f'  [PyPDF2] Page {page_num} error: {e}')
        except Exception as e:
            print(f'  [PyPDF2] File error: {e}')
        return text.strip()

    def extract_text_ocr(self, pdf_path: str) -> str:
        if not OCR_AVAILABLE:
            return ''
        text = ''
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            pages = convert_from_bytes(pdf_bytes, dpi=300)
            for i, page_img in enumerate(pages):
                try:
                    page_text = pytesseract.image_to_string(page_img, lang='eng')
                    text += page_text + '\n'
                except Exception as e:
                    print(f'  [OCR] Page {i+1} error: {e}')
        except Exception as e:
            print(f'  [OCR] Pipeline error: {e}')
        return text.strip()

    def extract_text(self, pdf_path: str) -> Tuple[str, str]:
        print(f'[ResumeParser] Reading: {Path(pdf_path).name}')
        if PDFPLUMBER_AVAILABLE:
            text = self.extract_text_pdfplumber(pdf_path)
            if len(text.strip()) >= 100:
                return text, 'pdfplumber'
            print('[ResumeParser] pdfplumber <100 chars, trying PyPDF2...')
        text = self.extract_text_pypdf2(pdf_path)
        if len(text.strip()) >= 100:
            return text, 'PyPDF2'
        print('[ResumeParser] PyPDF2 <100 chars, switching to OCR...')
        text   = self.extract_text_ocr(pdf_path)
        engine = 'OCR' if len(text.strip()) >= 50 else 'all_failed'
        return text, engine

    def _isolate_section(self, text: str, start_patterns: List[str],
                          end_patterns: List[str]) -> str:
        lines, collecting, section_lines, found = text.split('\n'), False, [], False
        for line in lines:
            stripped = line.strip()
            lower    = stripped.lower()
            if not collecting:
                for pat in start_patterns:
                    if re.search(pat, lower):
                        collecting = True
                        found      = True
                        break
                continue
            if collecting:
                is_end = any(re.search(pat, lower) and len(stripped) < 50
                             for pat in end_patterns)
                if is_end:
                    break
                if stripped:
                    section_lines.append(stripped)
        return '\n'.join(section_lines) if found else ''

    def _common_end_patterns(self) -> List[str]:
        return [
            r'^work\s*experience',       r'^professional\s*experience',
            r'^internship[s]?',          r'^experience',
            r'^projects?',               r'^technical\s*skills?',
            r'^skills?',                 r'^certifications?',
            r'^extra[\s\-]curricular',   r'^achievements?',
            r'^awards?',                 r'^publications?',
            r'^volunteer',               r'^activities',
            r'^references?',             r'^summary',
            r'^languages?',              r'^hobbies',
            r'^interests?',              r'^declaration',
        ]

    def extract_name(self, text: str) -> str:
        skip_words = [
            'resume', 'curriculum', 'vitae', 'cv', 'profile', 'summary',
            'objective', 'contact', 'email', 'phone', 'address', 'http',
            'linkedin', 'github', 'portfolio', 'page'
        ]
        for line in text.split('\n')[:10]:
            line = line.strip()
            if not line:
                continue
            if any(w in line.lower() for w in skip_words):
                continue
            if re.match(r'^[A-Za-z][A-Za-z\s\.]{2,50}$', line) \
                    and 1 <= len(line.split()) <= 5:
                return line.strip()
        return 'Name Not Found'

    def extract_email(self, text: str) -> str:
        m = re.findall(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
        return m[0] if m else 'Not Found'

    def extract_phone(self, text: str) -> str:
        for pattern in [
            r'\+91[\s\-]?[6-9]\d{9}', r'0[6-9]\d{9}',
            r'\b[6-9]\d{9}\b',        r'\(\+91\)\s*\d{10}',
        ]:
            m = re.findall(pattern, text)
            if m:
                return m[0].strip()
        return 'Not Found'

    def extract_education(self, text: str) -> str:
        edu_section = self._isolate_section(
            text,
            start_patterns=[
                r'^\s*education\s*$',
                r'^\s*academic\s*(background|qualifications?|details?)\s*$',
                r'^\s*educational\s*qualifications?\s*$',
            ],
            end_patterns=self._common_end_patterns()
        )
        if not edu_section.strip():
            return 'Not Found'

        lines         = edu_section.split('\n')
        section_lower = edu_section.lower()

        degree_hierarchy = [
            (r'ph\.?d|doctorate',                        'PhD'),
            (r'm\.tech|mtech',                           'M.Tech'),
            (r'm\.e\b',                                  'M.E'),
            (r'mba',                                     'MBA'),
            (r'm\.sc|msc\b',                             'M.Sc'),
            (r'mca\b',                                   'MCA'),
            (r'pgdm',                                    'PGDM'),
            (r"master'?s?",                              "Master's"),
            (r'b\.tech|btech',                           'B.Tech'),
            (r'b\.e\b',                                  'B.E'),
            (r'b\.sc|bsc\b',                             'B.Sc'),
            (r'bca\b',                                   'BCA'),
            (r'b\.com|bcom\b',                           'B.Com'),
            (r"bachelor'?s?",                            "Bachelor's"),
            (r'diploma',                                 'Diploma'),
            (r'12th|class\s*xii|hsc|higher\s*secondary', '12th Standard'),
            (r'10th|class\s*x\b|ssc|secondary\s*school', '10th Standard'),
        ]
        detected_degree = ''
        for pattern, label in degree_hierarchy:
            if re.search(pattern, section_lower):
                detected_degree = label
                break

        specialization = ''
        spec_patterns = [
            r'(?:b\.tech|btech|m\.tech|mtech|b\.e|m\.e|bachelor|master)'
            r'\s*\.?\s*in\s+([A-Za-z\s&\(\)\/]+?)(?:\s*[-–,\(]|$)',
            r'(?:b\.sc|bsc|m\.sc|msc)\s*\.?\s*in\s+([A-Za-z\s&]+?)'
            r'(?:\s*[-–,\(]|$)',
        ]
        for pat in spec_patterns:
            m = re.search(pat, section_lower)
            if m:
                raw = m.group(1).strip()
                raw = re.sub(r'\s*(and|with|from|at|the|a|an)\s*$', '', raw)
                if 2 < len(raw) < 70:
                    specialization = raw.title()
                    break

        grade = ''
        cgpa_m = re.search(
            r'(?:cgpa|gpa)\s*[:\-]?\s*(\d+\.\d+)\s*(?:/\s*(?:10|4))?',
            section_lower)
        if cgpa_m:
            grade = f'CGPA: {cgpa_m.group(1)}'
        else:
            pct_m = re.search(
                r'(?:percentage|marks)\s*[:\-]?\s*(\d+\.?\d*)\s*(?:/100|%)?|(\d+\.?\d*)\s*%',
                section_lower)
            if pct_m:
                grade = f'{pct_m.group(1) or pct_m.group(2)}%'

        institution    = ''
        institution_kw = ['university','institute','college','school',
                          'iit','nit','bits','vtu','academy','polytechnic']
        for line in lines:
            if any(kw in line.lower() for kw in institution_kw):
                clean = re.sub(
                    r'b\.?tech|m\.?tech|b\.?e\b|m\.?e\b|b\.?sc|m\.?sc'
                    r'|bca|mca|mba|phd|bachelor|master|diploma'
                    r'|cgpa.*|percentage.*|gpa.*|\d{4}.*',
                    '', line, flags=re.IGNORECASE).strip()
                clean = re.split(r'\t|\|', clean)[0].strip()
                clean = re.sub(r'\s+', ' ', clean).strip(' ,–-')
                if len(clean) > 4:
                    institution = clean
                    break

        duration = ''
        dm = re.search(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s*\d{4}|\d{4})'
            r'\s*[\–\-–—]\s*'
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s*\d{4}'
            r'|\d{4}|[Pp]resent|[Cc]urrent)',
            edu_section)
        if dm:
            duration = f'{dm.group(1).strip()}-{dm.group(2).strip()}'

        if not detected_degree:
            detected_degree = 'Degree Not Detected'
        degree_full = f'{detected_degree} in {specialization}' if specialization else detected_degree
        if grade:
            degree_full = f'{degree_full} ({grade})'
        if institution and duration:
            display = f'{degree_full}, {institution} ({duration})'
        elif institution:
            display = f'{degree_full}, {institution}'
        elif duration:
            display = f'{degree_full} ({duration})'
        else:
            display = degree_full
        return display

    def extract_skills(self, text: str) -> List[str]:
        common_ends = self._common_end_patterns()
        skills_section = self._isolate_section(
            text,
            start_patterns=[
                r'^\s*technical\s*skills?\s*$', r'^\s*skills?\s*$',
                r'^\s*core\s*competencies\s*$', r'^\s*key\s*skills?\s*$',
                r'^\s*technologies\s*$',
                r'^\s*tools?\s*(and\s*technologies?)?\s*$',
            ],
            end_patterns=common_ends + [r'^projects?']
        )
        projects_section = self._isolate_section(
            text,
            start_patterns=[
                r'^\s*projects?\s*$',            r'^\s*personal\s*projects?\s*$',
                r'^\s*academic\s*projects?\s*$', r'^\s*key\s*projects?\s*$',
            ],
            end_patterns=common_ends + [r'^technical\s*skills?', r'^skills?']
        )
        combined = (skills_section + '\n' + projects_section).strip() or text
        combined_lower = combined.lower()
        found = []
        for skill in self.skill_database:
            skill_lower = skill.lower()
            if len(skill_lower.split()) == 1:
                if re.search(r'\b' + re.escape(skill_lower) + r'\b', combined_lower):
                    found.append(skill)
            else:
                if skill_lower in combined_lower:
                    found.append(skill)
        return sorted(set(found))

    def _isolate_experience_section(self, text: str) -> str:
        return self._isolate_section(
            text,
            start_patterns=[
                r'^\s*work\s+experience\s*$',    r'^\s*professional\s+experience\s*$',
                r'^\s*employment\s+history\s*$', r'^\s*internship[s]?\s*$',
                r'^\s*work\s+history\s*$',       r'^\s*experience\s*$',
            ],
            end_patterns=[
                r'^education',           r'^projects?',
                r'^technical\s*skills?', r'^skills?',
                r'^certifications?',     r'^achievements?',
                r'^references?',         r'^languages?',
                r'^hobbies',             r'^declaration',
            ]
        )

    def extract_experience(self, text: str) -> dict:
        import datetime
        try:
            from dateutil import relativedelta as rdelta
        except ImportError:
            import subprocess as _sp, sys as _sys
            _sp.run([_sys.executable, '-m', 'pip', 'install', 'python-dateutil', '-q'])
            from dateutil import relativedelta as rdelta

        def make_result(total_months: float) -> dict:
            total_months = max(0, int(round(total_months)))
            y, m = total_months // 12, total_months % 12
            if y == 0 and m == 0:   display = '0 months'
            elif y == 0:            display = f'{m} month{"s" if m > 1 else ""}'
            elif m == 0:            display = f'{y} year{"s" if y > 1 else ""}'
            else:                   display = f'{y} year{"s" if y > 1 else ""} {m} month{"s" if m > 1 else ""}'
            return {'years': y, 'months': m, 'total_months': total_months, 'display': display}

        exp_section = self._isolate_experience_section(text)
        if not exp_section.strip():
            return make_result(0)

        for pattern, unit in [
            (r'(\d+\.?\d*)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', 'years'),
            (r'(\d+)\s*months?\s*(?:of\s*)?(?:experience|exp)',          'months'),
            (r'experience\s*(?:of\s*)?(\d+\.?\d*)\s*years?',            'years'),
        ]:
            matches = re.findall(pattern, exp_section.lower())
            if matches:
                try:
                    val = float(matches[0])
                    return make_result(val * 12 if unit == 'years' else val)
                except ValueError:
                    pass

        month_map = {
            'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
            'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
            'january':1,'february':2,'march':3,'april':4,'june':6,
            'july':7,'august':8,'september':9,'october':10,
            'november':11,'december':12,
        }
        now = datetime.datetime.now()

        def parse_date(s: str):
            s = s.strip().lower()
            if s in ('present','current','now','till date','ongoing'):
                return now
            m = re.match(r'([a-z]+)\.?\s+(\d{4})', s)
            if m:
                return datetime.datetime(int(m.group(2)), month_map.get(m.group(1), 1), 1)
            m = re.match(r'^(\d{4})$', s)
            if m:
                return datetime.datetime(int(m.group(1)), 1, 1)
            return None

        ranges = re.findall(
            r'([A-Za-z]+\.?\s+\d{4}|\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)'
            r'\s*[\–\-–—]\s*'
            r'([A-Za-z]+\.?\s+\d{4}|\d{4}|[Pp]resent|[Cc]urrent|[Nn]ow)',
            exp_section
        )
        total_months = 0
        for s1, s2 in ranges:
            d1, d2 = parse_date(s1), parse_date(s2)
            if d1 and d2 and d2 >= d1:
                try:
                    diff          = rdelta.relativedelta(d2, d1)
                    total_months += diff.years * 12 + diff.months
                except Exception:
                    pass
        return make_result(total_months)

    def extract_experience_blocks(self, text: str) -> List[str]:
        blocks, in_section, current_block = [], False, []
        hdrs = [
            r'(work\s*experience|professional\s*experience|employment|experience)',
            r'(projects?|internship|work\s*history)',
        ]
        for line in text.split('\n'):
            ll = line.lower().strip()
            if any(re.search(h, ll) for h in hdrs):
                in_section = True
                if current_block:
                    blocks.append(' '.join(current_block))
                    current_block = []
                continue
            if in_section and line.strip():
                current_block.append(line.strip())
                if len(current_block) > 50:
                    blocks.append(' '.join(current_block))
                    current_block = []
                    in_section    = False
        if current_block:
            blocks.append(' '.join(current_block))
        return blocks or [text[:2000]]

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        start  = time.time()
        result = {
            'file':               Path(pdf_path).name,
            'name':               'Unknown',
            'email':              'Not Found',
            'phone':              'Not Found',
            'experience_years':   0.0,
            'experience_display': '0 months',
            'education_display':  'Not Found',
            'skills':             [],
            'experience_blocks':  [],
            'raw_text':           '',
            'parse_status':       'success',
            'parse_engine':       'unknown',
            'parse_confidence':   0,
            'parse_time_sec':     0.0,
        }
        try:
            raw_text, engine = self.extract_text(pdf_path)
            result['parse_engine'] = engine
            if not raw_text.strip():
                result['parse_status'] = 'empty_pdf'
                self.auditor.audit(result)
                return result
            raw_text                     = raw_text.encode('utf-8', errors='replace').decode('utf-8')
            exp_data                     = self.extract_experience(raw_text)
            result['raw_text']           = raw_text
            result['name']               = self.extract_name(raw_text)
            result['email']              = self.extract_email(raw_text)
            result['phone']              = self.extract_phone(raw_text)
            result['experience_years']   = round(exp_data['total_months'] / 12, 2)
            result['experience_display'] = exp_data['display']
            result['education_display']  = self.extract_education(raw_text)
            result['skills']             = self.extract_skills(raw_text)
            result['experience_blocks']  = self.extract_experience_blocks(raw_text)
        except Exception as e:
            result['parse_status'] = f'error: {str(e)}'
            print(f'[ResumeParser] ERROR: {e}')
            traceback.print_exc()

        result['parse_time_sec'] = round(time.time() - start, 2)
        audit = self.auditor.audit(result)
        result['parse_confidence'] = audit['confidence']
        return result


# ── EmbeddingsService ─────────────────────────────────────────

class EmbeddingsService:

    BASE_MODEL     = 'all-MiniLM-L6-v2'
    FINETUNED_PATH = './finetuned_sbert_recruitment'

    def __init__(self):
        self.model         = None
        self.fallback_mode = False
        self.model_name    = ''
        self._load_model()

    def _load_model(self):
        if not EMBEDDINGS_AVAILABLE:
            print('[EmbeddingsService] SentenceTransformers unavailable — TF-IDF fallback.')
            self.fallback_mode = True
            return

        ft_path = self.FINETUNED_PATH
        if (os.path.exists(ft_path) and
                os.path.exists(os.path.join(ft_path, 'config.json'))):
            try:
                self.model      = SentenceTransformer(ft_path)
                self.model_name = 'Fine-tuned SBERT (Recruitment Domain)'
                print('[EmbeddingsService] ✅ Fine-tuned model loaded.')
                return
            except Exception as e:
                print(f'[EmbeddingsService] Fine-tuned load failed: {e}')

        try:
            self.model      = SentenceTransformer(self.BASE_MODEL)
            self.model_name = f'Base SBERT ({self.BASE_MODEL})'
            print('[EmbeddingsService] ✅ Base model loaded.')
        except Exception as e:
            print(f'[EmbeddingsService] Model load failed: {e} — TF-IDF fallback.')
            self.fallback_mode = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not isinstance(texts, list):
            texts = [texts]
        texts = [str(t) if t else ' ' for t in texts]
        if self.fallback_mode or self.model is None:
            return self._tfidf_encode(texts)
        try:
            return self.model.encode(
                texts, convert_to_numpy=True,
                show_progress_bar=False, batch_size=32)
        except Exception as e:
            print(f'[EmbeddingsService] Encoding error: {e}')
            return self._tfidf_encode(texts)

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            if len(texts) == 1:
                texts = texts + ['placeholder fallback text']
            vec    = TfidfVectorizer(max_features=512, stop_words='english')
            matrix = vec.fit_transform(texts).toarray()
            return matrix[:len(texts)-1] if len(texts) > 1 else matrix
        except Exception:
            return np.zeros((len(texts), 128))

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        try:
            if emb1.ndim == 1: emb1 = emb1.reshape(1, -1)
            if emb2.ndim == 1: emb2 = emb2.reshape(1, -1)
            return float(np.clip(cosine_similarity(emb1, emb2)[0][0], 0.0, 1.0))
        except Exception:
            return 0.0

    def batch_similarity(self, query_emb: np.ndarray,
                          corpus_embs: np.ndarray) -> np.ndarray:
        try:
            if query_emb.ndim == 1:
                query_emb = query_emb.reshape(1, -1)
            return np.clip(cosine_similarity(query_emb, corpus_embs)[0], 0.0, 1.0)
        except Exception:
            return np.zeros(len(corpus_embs))

    def get_model_info(self) -> str:
        return 'TF-IDF Fallback' if self.fallback_mode else self.model_name


# ── MatcherService ────────────────────────────────────────────

class MatcherService:

    WEIGHTS = {
        'semantic':   0.40,
        'skill':      0.35,
        'experience': 0.20,
        'education':  0.05,
    }

    RECOMMENDATION_THRESHOLDS = [
        (80, 'Excellent', '🟢'),
        (60, 'Good',      '🟡'),
        (40, 'Fair',      '🟠'),
        (0,  'Poor',      '🔴'),
    ]

    EDUCATION_TIERS = {
        'phd': 5, 'ph.d': 5,
        'm.tech': 4, 'mtech': 4, 'm.e': 4, 'mba': 4,
        'm.sc': 4, 'msc': 4, "master's": 4, 'masters': 4,
        'b.tech': 3, 'btech': 3, 'b.e': 3, 'b.sc': 3,
        'bsc': 3, "bachelor's": 3, 'bachelors': 3, 'bca': 3,
        'diploma': 2, '12th': 1, '10th': 0,
    }

    def __init__(self, emb_service: EmbeddingsService):
        self.emb = emb_service

    def score_skills(self, candidate_skills, required_skills,
                     nice_to_have=None) -> Dict:
        if nice_to_have is None:
            nice_to_have = []
        c_set = set(s.lower().strip() for s in candidate_skills)
        r_set = set(s.lower().strip() for s in required_skills)
        n_set = set(s.lower().strip() for s in nice_to_have)

        matched_req  = r_set & c_set
        missing_req  = r_set - c_set
        matched_nice = n_set & c_set
        extra        = c_set - r_set - n_set

        req_ratio  = len(matched_req) / max(len(r_set), 1)
        nice_bonus = (len(matched_nice) / max(len(n_set), 1)) * 0.20 if n_set else 0
        score      = min((req_ratio + nice_bonus) * 100, 100)

        return {
            'score':                round(score, 2),
            'matched_required':     sorted(matched_req),
            'missing_required':     sorted(missing_req),
            'matched_nice_to_have': sorted(matched_nice),
            'extra_skills':         sorted(extra),
            'required_match_ratio': round(req_ratio * 100, 1),
        }

    def score_experience(self, candidate_exp: float,
                          req_min: float, req_max: float = None) -> Dict:
        if req_max is None:
            req_max = req_min + 5
        if req_min <= candidate_exp <= req_max:
            score, verdict = 100.0, 'Ideal experience match'
        elif candidate_exp < req_min:
            gap     = req_min - candidate_exp
            score   = max(100 - gap * 15, 0)
            verdict = f'Underqualified by {round(gap, 1)} year(s)'
        else:
            gap     = candidate_exp - req_max
            score   = max(100 - gap * 8, 60)
            verdict = f'Overqualified by {round(gap, 1)} year(s)'
        return {
            'score': round(score, 2), 'candidate_experience': candidate_exp,
            'required_min': req_min, 'required_max': req_max, 'verdict': verdict,
        }

    def score_education(self, education_display: str,
                         job_description: str) -> Dict:
        edu_lower = education_display.lower()
        jd_lower  = job_description.lower()
        candidate_tier = 0
        for keyword, tier in self.EDUCATION_TIERS.items():
            if keyword in edu_lower:
                candidate_tier = tier
                break
        required_tier = 2
        if any(w in jd_lower for w in ['phd', 'ph.d', 'doctorate']):
            required_tier = 5
        elif any(w in jd_lower for w in ['m.tech', 'mtech', 'masters', "master's", 'mba']):
            required_tier = 4
        elif any(w in jd_lower for w in ['b.tech', 'btech', 'bachelor', 'degree']):
            required_tier = 3
        if candidate_tier >= required_tier:
            score, verdict = 100.0, 'Education meets or exceeds requirement'
        elif candidate_tier == required_tier - 1:
            score, verdict = 60.0, 'Education slightly below requirement'
        else:
            score   = max(20.0, (candidate_tier / max(required_tier, 1)) * 100)
            verdict = 'Education below requirement'
        return {
            'score': round(score, 2), 'candidate_tier': candidate_tier,
            'required_tier': required_tier, 'verdict': verdict,
        }

    def score_semantic(self, exp_blocks: List[str],
                        job_description: str) -> Dict:
        if not exp_blocks or not job_description.strip():
            return {'score': 0.0, 'max_similarity': 0.0,
                    'mean_similarity': 0.0, 'num_blocks_analyzed': 0}
        try:
            jd_emb     = self.emb.encode([job_description])[0]
            block_embs = self.emb.encode(exp_blocks)
            sims       = self.emb.batch_similarity(jd_emb, block_embs)
            max_sim    = float(np.max(sims))
            mean_sim   = float(np.mean(sims))
            score      = (0.70 * max_sim + 0.30 * mean_sim) * 100
            return {
                'score':               round(score, 2),
                'max_similarity':      round(max_sim, 4),
                'mean_similarity':     round(mean_sim, 4),
                'num_blocks_analyzed': len(exp_blocks),
            }
        except Exception as e:
            print(f'[MatcherService] Semantic error: {e}')
            return {'score': 0.0, 'max_similarity': 0.0,
                    'mean_similarity': 0.0, 'num_blocks_analyzed': 0}

    def get_recommendation(self, score: float) -> Tuple[str, str]:
        for threshold, label, emoji in self.RECOMMENDATION_THRESHOLDS:
            if score >= threshold:
                return label, emoji
        return 'Poor', '🔴'

    def generate_feedback(self, parsed, skill_r, exp_r, sem_r, edu_r,
                           final_score, recommendation) -> str:
        name        = parsed.get('name', 'Candidate')
        matched     = skill_r.get('matched_required', [])
        missing     = skill_r.get('missing_required', [])
        nice        = skill_r.get('matched_nice_to_have', [])
        exp_display = parsed.get('experience_display', '0 months')
        edu_display = parsed.get('education_display', 'Not Found')

        lines = [
            '📋 CANDIDATE EVALUATION REPORT',
            '=' * 60,
            f'Candidate  : {name}',
            f'Email      : {parsed.get("email", "N/A")}',
            f'Phone      : {parsed.get("phone", "N/A")}',
            f'Experience : {exp_display}',
            f'Education  : {edu_display}',
            '',
            f'🏆 FINAL SCORE    : {final_score:.1f} / 100',
            f'📌 RECOMMENDATION : {recommendation}',
            '',
            '📊 SCORING BREAKDOWN:',
            f'  Semantic  : {sem_r["score"]:.1f}/100 × 40% = {sem_r["score"]*0.40:.1f}',
            f'  Skill     : {skill_r["score"]:.1f}/100 × 35% = {skill_r["score"]*0.35:.1f}',
            f'  Experience: {exp_r["score"]:.1f}/100 × 20% = {exp_r["score"]*0.20:.1f}',
            f'  Education : {edu_r["score"]:.1f}/100 ×  5% = {edu_r["score"]*0.05:.1f}',
            '',
            f'✅ MATCHED SKILLS ({len(matched)}/{len(matched)+len(missing)}):',
            f'  {", ".join(matched) if matched else "None"}',
            '',
            f'❌ MISSING SKILLS ({len(missing)}):',
            f'  {", ".join(missing) if missing else "None — All matched!"}',
            '',
            f'⭐ NICE-TO-HAVE MATCHED ({len(nice)}):',
            f'  {", ".join(nice) if nice else "None"}',
            '',
            f'📈 EXPERIENCE: {exp_r["verdict"]}',
            f'   Candidate: {exp_display} | Required: {exp_r["required_min"]}–{exp_r["required_max"]} yrs',
            '',
            f'🎓 EDUCATION: {edu_r["verdict"]}  ({edu_display})',
            '',
            f'🔍 SEMANTIC: max={sem_r["max_similarity"]:.4f}  mean={sem_r["mean_similarity"]:.4f}  '
            f'blocks={sem_r.get("num_blocks_analyzed", 0)}',
        ]

        lines += ['', '=' * 60, '📝 SUMMARY:']
        if   final_score >= 80:
            lines.append(f'  {name} — EXCELLENT fit. Recommend for interview.')
        elif final_score >= 60:
            lines.append(f'  {name} — GOOD candidate with minor gaps.')
        elif final_score >= 40:
            lines.append(f'  {name} — FAIR match. Consider for junior role.')
        else:
            lines.append(f'  {name} — POOR fit. Significant mismatch.')
        return '\n'.join(lines)

    def analyze_for_jobseeker(self, parsed: Dict, job_description: str,
                               required_skills: List[str],
                               nice_to_have: List[str]) -> str:
        skill_r     = self.score_skills(parsed.get('skills', []),
                                        required_skills, nice_to_have)
        sem_r       = self.score_semantic(parsed.get('experience_blocks', []),
                                          job_description)
        edu_r       = self.score_education(parsed.get('education_display', ''),
                                           job_description)
        name        = parsed.get('name', 'You')
        matched     = skill_r['matched_required']
        missing     = skill_r['missing_required']
        nice_match  = skill_r['matched_nice_to_have']
        extra       = list(skill_r['extra_skills'])
        skill_score = skill_r['score']
        sem_score   = sem_r['score']
        edu_score   = edu_r['score']
        overall     = round(0.40*sem_score + 0.35*skill_score
                            + 0.20*50 + 0.05*edu_score, 1)

        fit_label = ('🟢 Strong Fit'        if overall >= 75 else
                     '🟡 Moderate Fit'      if overall >= 50 else
                     '🔴 Needs Improvement')

        suggestions = []
        if missing:
            suggestions.append('  📚 Skills to learn:')
            for s in missing[:6]:
                suggestions.append(f'     • {s.title()} — Coursera, YouTube, or official docs.')
        if nice_match:
            suggestions.append(
                f'\n  ⭐ Nice-to-have you already have: {", ".join(nice_match)}. '
                f'Highlight in your cover letter!')
        if extra:
            suggestions.append(f'\n  💼 Extra skills: {", ".join(extra[:5])}. Mention if relevant.')
        if   skill_score >= 80:
            suggestions.append('\n  ✅ Strong skill profile — apply confidently!')
        elif skill_score >= 50:
            suggestions.append('\n  ⚠️  Core requirements met but gaps exist.')
        else:
            suggestions.append('\n  🚨 Significant gaps — 2–3 months upskilling recommended.')

        lines = [
            f'👤 RESUME–JD ANALYSIS FOR: {name}',
            '=' * 60,
            f'📊 OVERALL FIT SCORE : {overall:.1f} / 100',
            f'📌 FIT LEVEL         : {fit_label}',
            '',
            '📊 SCORE BREAKDOWN:',
            f'  Semantic : {sem_score:.1f}/100 × 40%',
            f'  Skill    : {skill_score:.1f}/100 × 35%',
            f'  Education: {edu_score:.1f}/100 ×  5%',
            '',
            f'✅ SKILLS YOU HAVE ({len(matched)}/{len(required_skills)}):',
            f'  {", ".join(matched) if matched else "None detected"}',
            '',
            f'❌ SKILLS MISSING ({len(missing)}):',
            f'  {", ".join(missing) if missing else "🎉 All required skills matched!"}',
            '',
            f'⭐ NICE-TO-HAVE YOU HAVE ({len(nice_match)}):',
            f'  {", ".join(nice_match) if nice_match else "None"}',
            '',
            f'🎓 EDUCATION: {edu_r["verdict"]}',
            '',
            '💡 PERSONALISED SUGGESTIONS:',
        ]
        lines.extend(suggestions)
        lines += ['', '=' * 60, '📝 SUMMARY:']
        if   overall >= 75: lines.append(f'  {name} — STRONG match, apply confidently!')
        elif overall >= 50: lines.append(f'  {name} — MODERATE fit, work on missing skills first.')
        else:               lines.append(f'  {name} — Significant gaps, upskill before applying.')
        return '\n'.join(lines)

    def match(self, parsed_resume, job_title, job_description,
              required_skills, nice_to_have, min_exp, max_exp) -> Dict:
        start   = time.time()
        skill_r = self.score_skills(parsed_resume.get('skills', []),
                                    required_skills, nice_to_have)
        exp_r   = self.score_experience(parsed_resume.get('experience_years', 0),
                                        min_exp, max_exp)
        sem_r   = self.score_semantic(parsed_resume.get('experience_blocks', []),
                                      job_description)
        edu_r   = self.score_education(parsed_resume.get('education_display', ''),
                                       job_description)

        final = round(min(max(
            self.WEIGHTS['semantic']   * sem_r['score']   +
            self.WEIGHTS['skill']      * skill_r['score'] +
            self.WEIGHTS['experience'] * exp_r['score']   +
            self.WEIGHTS['education']  * edu_r['score'],
        0), 100), 2)

        rec_label, rec_emoji = self.get_recommendation(final)
        recommendation       = f'{rec_emoji} {rec_label}'
        feedback             = self.generate_feedback(
            parsed_resume, skill_r, exp_r, sem_r, edu_r, final, recommendation)

        matched  = skill_r['matched_required']
        missing  = skill_r['missing_required']

        return {
            'name':                    parsed_resume.get('name', 'Unknown'),
            'email':                   parsed_resume.get('email', 'N/A'),
            'phone':                   parsed_resume.get('phone', 'N/A'),
            'experience_years':        parsed_resume.get('experience_years', 0),
            'experience_display':      parsed_resume.get('experience_display', '0 months'),
            'education_display':       parsed_resume.get('education_display', 'Not Found'),
            'skills_found':            f'{len(matched)} ({", ".join(sorted(matched))})' if matched else '0 (none matched)',
            'skills_missing':          f'{len(missing)} ({", ".join(sorted(missing))})' if missing else '0 ✅',
            'final_score':             final,
            'recommendation':          recommendation,
            'semantic_score':          sem_r['score'],
            'skill_score':             skill_r['score'],
            'experience_score':        exp_r['score'],
            'education_score':         edu_r['score'],
            'matched_required_skills': skill_r['matched_required'],
            'missing_required_skills': skill_r['missing_required'],
            'matched_nice_skills':     skill_r['matched_nice_to_have'],
            'extra_skills':            skill_r['extra_skills'],
            'experience_verdict':      exp_r['verdict'],
            'education_verdict':       edu_r['verdict'],
            'max_semantic_similarity': sem_r['max_similarity'],
            'feedback_report':         feedback,
            'file':                    parsed_resume.get('file', ''),
            'match_time_sec':          round(time.time() - start, 2),
        }

    def rank_batch(self, parsed_resumes, job_title, job_description,
                   required_skills, nice_to_have, min_exp, max_exp) -> pd.DataFrame:
        print(f'[MatcherService] Ranking {len(parsed_resumes)} resumes...')
        results = []
        for i, resume in enumerate(parsed_resumes):
            print(f'  [{i+1}/{len(parsed_resumes)}] {resume.get("name","Unknown")}')
            results.append(self.match(
                resume, job_title, job_description,
                required_skills, nice_to_have, min_exp, max_exp))
        df            = pd.DataFrame(results).sort_values(
            'final_score', ascending=False).reset_index(drop=True)
        df.index      = df.index + 1
        df.index.name = 'Rank'
        return df


# ── Singletons ────────────────────────────────────────────────

parser             = ResumeParser()
embeddings_service = EmbeddingsService()
matcher            = MatcherService(embeddings_service)

print('✅ model.py ready — parser, embeddings_service, matcher initialized.')