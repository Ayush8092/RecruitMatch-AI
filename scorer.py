# ============================================================
# scorer.py
# EmbeddingsService  — loads fine-tuned SBERT (or falls back)
# MatcherService     — hybrid weighted scoring
#
# Score = 0.40 × semantic  (fine-tuned SBERT)
#       + 0.35 × skill_match
#       + 0.20 × experience_match
#       + 0.05 × education_match
# ============================================================

import os
import time
import numpy as np
from typing import Dict, List, Tuple

from utils import EMBEDDINGS_AVAILABLE

if EMBEDDINGS_AVAILABLE:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity


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
            self.fallback_mode = True
            return

        ft_path = self.FINETUNED_PATH
        if (os.path.exists(ft_path) and
                os.path.exists(os.path.join(ft_path, 'config.json'))):
            try:
                self.model      = SentenceTransformer(ft_path)
                self.model_name = 'Fine-tuned SBERT (Recruitment Domain)'
                print(f'[EmbeddingsService] Fine-tuned model loaded from {ft_path}')
                return
            except Exception as e:
                print(f'[EmbeddingsService] Fine-tuned load failed: {e} — using base')

        try:
            self.model      = SentenceTransformer(self.BASE_MODEL)
            self.model_name = f'Base SBERT ({self.BASE_MODEL})'
            print(f'[EmbeddingsService] Base model loaded: {self.BASE_MODEL}')
        except Exception as e:
            print(f'[EmbeddingsService] Base model failed: {e} — TF-IDF fallback')
            self.fallback_mode = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not isinstance(texts, list):
            texts = [texts]
        texts = [str(t) if t else ' ' for t in texts]
        if self.fallback_mode or self.model is None:
            return self._tfidf_encode(texts)
        try:
            return self.model.encode(
                texts,
                convert_to_numpy  = True,
                show_progress_bar = False,
                batch_size        = 32,
            )
        except Exception:
            return self._tfidf_encode(texts)

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            if len(texts) == 1:
                texts = texts + ['placeholder fallback text']
            vec    = TfidfVectorizer(max_features=512, stop_words='english')
            matrix = vec.fit_transform(texts).toarray()
            return matrix[:len(texts) - 1] if len(texts) > 1 else matrix
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
        (80, 'Excellent', 'green'),
        (60, 'Good',      'yellow'),
        (40, 'Fair',      'orange'),
        (0,  'Poor',      'red'),
    ]

    EDUCATION_TIERS = {
        'phd': 5, 'ph.d': 5,
        'm.tech': 4, 'mtech': 4, 'm.e': 4, 'mba': 4,
        'm.sc': 4, 'msc': 4, "master's": 4, 'masters': 4,
        'b.tech': 3, 'btech': 3, 'b.e': 3, 'b.sc': 3,
        'bsc': 3, "bachelor's": 3, 'bachelors': 3, 'bca': 3,
        'diploma': 2,
        '12th': 1, '10th': 0,
    }

    def __init__(self, emb_service: EmbeddingsService):
        self.emb = emb_service

    # ── Skill scoring (35%) ───────────────────────────────────
    def score_skills(self, candidate_skills: List[str],
                     required_skills: List[str],
                     nice_to_have: List[str] = None) -> Dict:
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

    # ── Experience scoring (20%) ──────────────────────────────
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
            'score':                round(score, 2),
            'candidate_experience': candidate_exp,
            'required_min':         req_min,
            'required_max':         req_max,
            'verdict':              verdict,
        }

    # ── Education scoring (5%) ────────────────────────────────
    def score_education(self, education_display: str, job_description: str) -> Dict:
        edu_lower = education_display.lower()
        jd_lower  = job_description.lower()

        candidate_tier = 0
        for keyword, tier in self.EDUCATION_TIERS.items():
            if keyword in edu_lower:
                candidate_tier = tier
                break

        required_tier = 2
        if any(w in jd_lower for w in ['phd', 'ph.d', 'doctorate', 'doctoral']):
            required_tier = 5
        elif any(w in jd_lower for w in ['m.tech', 'mtech', 'masters',
                                           "master's", 'mba', 'm.sc',
                                           'postgraduate', 'pg']):
            required_tier = 4
        elif any(w in jd_lower for w in ['b.tech', 'btech', 'bachelor',
                                           "bachelor's", 'b.e',
                                           'undergraduate', 'degree']):
            required_tier = 3

        if candidate_tier >= required_tier:
            score, verdict = 100.0, 'Education meets or exceeds requirement'
        elif candidate_tier == required_tier - 1:
            score, verdict = 60.0, 'Education slightly below requirement'
        else:
            score   = max(20.0, (candidate_tier / max(required_tier, 1)) * 100)
            verdict = 'Education below requirement'

        return {
            'score':          round(score, 2),
            'candidate_tier': candidate_tier,
            'required_tier':  required_tier,
            'verdict':        verdict,
        }

    # ── Semantic scoring (40%) ────────────────────────────────
    def score_semantic(self, exp_blocks: List[str], job_description: str) -> Dict:
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
        except Exception:
            return {'score': 0.0, 'max_similarity': 0.0,
                    'mean_similarity': 0.0, 'num_blocks_analyzed': 0}

    def get_recommendation(self, score: float) -> Dict:
        for threshold, label, color in self.RECOMMENDATION_THRESHOLDS:
            if score >= threshold:
                return {'label': label, 'color': color}
        return {'label': 'Poor', 'color': 'red'}

    # ── Single resume match ───────────────────────────────────
    def match(self, parsed_resume: Dict, job_title: str,
              job_description: str, required_skills: List[str],
              nice_to_have: List[str], min_exp: float,
              max_exp: float) -> Dict:
        start   = time.time()
        skill_r = self.score_skills(
            parsed_resume.get('skills', []), required_skills, nice_to_have)
        exp_r   = self.score_experience(
            parsed_resume.get('experience_years', 0), min_exp, max_exp)
        sem_r   = self.score_semantic(
            parsed_resume.get('experience_blocks', []), job_description)
        edu_r   = self.score_education(
            parsed_resume.get('education_display', ''), job_description)

        final = round(min(max(
            self.WEIGHTS['semantic']   * sem_r['score']   +
            self.WEIGHTS['skill']      * skill_r['score'] +
            self.WEIGHTS['experience'] * exp_r['score']   +
            self.WEIGHTS['education']  * edu_r['score'],
        0), 100), 2)

        rec = self.get_recommendation(final)

        return {
            'candidate': {
                'name':               parsed_resume.get('name', 'Unknown'),
                'email':              parsed_resume.get('email', 'N/A'),
                'phone':              parsed_resume.get('phone', 'N/A'),
                'experience_years':   parsed_resume.get('experience_years', 0),
                'experience_display': parsed_resume.get('experience_display', '0 months'),
                'education_display':  parsed_resume.get('education_display', 'Not Found'),
                'parse_engine':       parsed_resume.get('parse_engine', 'unknown'),
                'parse_confidence':   parsed_resume.get('parse_confidence', 0),
            },
            'scores': {
                'final':      final,
                'semantic':   sem_r['score'],
                'skill':      skill_r['score'],
                'experience': exp_r['score'],
                'education':  edu_r['score'],
            },
            'recommendation': rec,
            'skills': {
                'matched_required':     skill_r['matched_required'],
                'missing_required':     skill_r['missing_required'],
                'matched_nice_to_have': skill_r['matched_nice_to_have'],
                'extra_skills':         sorted(skill_r['extra_skills'])[:10],
                'required_match_ratio': skill_r['required_match_ratio'],
            },
            'experience': {
                'verdict':      exp_r['verdict'],
                'required_min': exp_r['required_min'],
                'required_max': exp_r['required_max'],
            },
            'education': {
                'verdict': edu_r['verdict'],
            },
            'semantic': {
                'max_similarity':      sem_r['max_similarity'],
                'mean_similarity':     sem_r['mean_similarity'],
                'num_blocks_analyzed': sem_r.get('num_blocks_analyzed', 0),
            },
            'match_time_sec': round(time.time() - start, 2),
            'file':           parsed_resume.get('file', ''),
        }

    # ── Batch ranking ─────────────────────────────────────────
    def rank_batch(self, parsed_resumes: List[Dict], job_title: str,
                   job_description: str, required_skills: List[str],
                   nice_to_have: List[str], min_exp: float,
                   max_exp: float) -> List[Dict]:
        results = [
            self.match(r, job_title, job_description,
                       required_skills, nice_to_have, min_exp, max_exp)
            for r in parsed_resumes
        ]
        results.sort(key=lambda x: x['scores']['final'], reverse=True)
        for rank, r in enumerate(results, 1):
            r['rank'] = rank
        return results

    # ── Job-seeker analysis ───────────────────────────────────
    def analyze_for_jobseeker(self, parsed: Dict, job_description: str,
                               required_skills: List[str],
                               nice_to_have: List[str]) -> Dict:
        skill_r = self.score_skills(
            parsed.get('skills', []), required_skills, nice_to_have)
        sem_r   = self.score_semantic(
            parsed.get('experience_blocks', []), job_description)
        edu_r   = self.score_education(
            parsed.get('education_display', ''), job_description)

        skill_score = skill_r['score']
        sem_score   = sem_r['score']
        edu_score   = edu_r['score']
        overall     = round(
            0.40 * sem_score + 0.35 * skill_score +
            0.20 * 50        + 0.05 * edu_score, 1)

        if overall >= 75:
            fit_level = 'Strong Fit'
        elif overall >= 50:
            fit_level = 'Moderate Fit'
        else:
            fit_level = 'Needs Improvement'

        return {
            'candidate_name': parsed.get('name', 'Unknown'),
            'overall_score':  overall,
            'fit_level':      fit_level,
            'scores': {
                'semantic':  sem_score,
                'skill':     skill_score,
                'education': edu_score,
            },
            'skills': {
                'matched_required':     skill_r['matched_required'],
                'missing_required':     skill_r['missing_required'],
                'matched_nice_to_have': skill_r['matched_nice_to_have'],
                'required_match_ratio': skill_r['required_match_ratio'],
            },
            'education_verdict': edu_r['verdict'],
            'semantic': {
                'relevance': (
                    'High'     if sem_score >= 60 else
                    'Moderate' if sem_score >= 40 else 'Low'
                ),
                'score': sem_score,
            },
        }


# ── Singletons ────────────────────────────────────────────────
embeddings_service = EmbeddingsService()
matcher            = MatcherService(embeddings_service)