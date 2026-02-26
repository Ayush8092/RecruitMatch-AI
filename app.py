# ============================================================
# app.py
# Gradio UI — AI Resume Ranking System
# HuggingFace Spaces entry point
# ============================================================

import json
import gradio as gr
from model import parser, embeddings_service, matcher

# ── Sample defaults ───────────────────────────────────────────

SAMPLE_JD = """We are looking for a Senior Machine Learning Engineer to join our AI team.
You will design, build, and deploy production-grade ML systems.

Responsibilities:
- Develop NLP models for text classification, NER, and semantic search
- Build and maintain ML pipelines using MLflow and Airflow
- Work with large datasets using Spark, Pandas, and distributed computing
- Design REST APIs using FastAPI or Flask

Requirements:
- 3+ years of hands-on ML experience
- Strong Python with TensorFlow or PyTorch
- Experience with NLP frameworks (HuggingFace, spaCy, NLTK)
- Cloud platform experience (AWS, GCP, or Azure)
- SQL and NoSQL database experience
"""

SAMPLE_REQUIRED_SKILLS = 'python, machine learning, tensorflow, pytorch, nlp, sql, docker'
SAMPLE_NICE_SKILLS     = 'kubernetes, mlflow, airflow, spark, huggingface, fastapi'


# ── Callbacks ─────────────────────────────────────────────────

def parse_resume_ui(pdf_files):
    if not pdf_files:
        return 'No files uploaded.', '{}'
    parser.auditor.reset()
    all_parsed, display_lines = [], []
    for pdf_file in pdf_files:
        path   = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
        parsed = parser.parse(path)
        all_parsed.append(parsed)
        display_lines += [
            '─' * 62,
            f'📄 File            : {parsed["file"]}',
            f'👤 Name            : {parsed["name"]}',
            f'📧 Email           : {parsed["email"]}',
            f'📱 Phone           : {parsed["phone"]}',
            f'🕐 Experience      : {parsed["experience_display"]}',
            f'🎓 Education       : {parsed["education_display"]}',
            '',
            f'🛠️  Skills ({len(parsed["skills"])}):',
            f'   {", ".join(parsed["skills"]) if parsed["skills"] else "None detected"}',
            f'🔧 Parse Engine    : {parsed.get("parse_engine", "unknown")}',
            f'📊 Parse Confidence: {parsed.get("parse_confidence", 0)}/100',
            f'⏱️  Parse Time      : {parsed["parse_time_sec"]}s',
            '',
        ]
    display_lines += ['', parser.auditor.batch_report()]
    json_output = json.dumps([
        {k: v for k, v in p.items() if k not in ['raw_text', 'experience_blocks']}
        for p in all_parsed
    ], indent=2)
    return '\n'.join(display_lines), json_output


def match_resumes_ui(pdf_files, job_title, job_description,
                     required_skills_str, nice_skills_str, min_exp, max_exp):
    if not pdf_files:
        return None, 'No files uploaded.'
    if not job_description.strip():
        return None, 'Please enter a job description.'
    if not required_skills_str.strip():
        return None, 'Please enter at least one required skill.'

    parser.auditor.reset()
    required_skills = [s.strip().lower() for s in required_skills_str.split(',') if s.strip()]
    nice_skills     = [s.strip().lower() for s in nice_skills_str.split(',') if s.strip()]

    parsed_list = []
    for pdf_file in pdf_files:
        path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
        parsed_list.append(parser.parse(path))

    df = matcher.rank_batch(
        parsed_list,
        job_title       = job_title,
        job_description = job_description,
        required_skills = required_skills,
        nice_to_have    = nice_skills,
        min_exp         = float(min_exp),
        max_exp         = float(max_exp),
    )

    display_cols = [
        'name', 'final_score', 'recommendation',
        'semantic_score', 'skill_score', 'experience_score', 'education_score',
        'skills_found', 'skills_missing', 'experience_display', 'education_display',
    ]
    display_df         = df[display_cols].copy()
    display_df.columns = [
        'Candidate', 'Final Score', 'Recommendation',
        'Semantic', 'Skill', 'Exp', 'Edu',
        'Skills Matched', 'Skills Missing', 'Experience', 'Education',
    ]

    summary = [
        f'🎯 JOB MATCHING RESULTS — {job_title}',
        '=' * 62,
        f'Total Resumes  : {len(df)}',
        f'Required Skills: {", ".join(required_skills[:6])}',
        f'Exp Range      : {min_exp}–{max_exp} years',
        '',
        '📊 WEIGHTS: Semantic×40% | Skill×35% | Exp×20% | Edu×5%',
        f'🤖 MODEL  : {embeddings_service.get_model_info()}',
        '',
    ]
    for rank, row in df.iterrows():
        summary += [
            f'Rank #{rank}: {row["name"]}  |  Score: {row["final_score"]:.1f}  |  {row["recommendation"]}',
            f'  Semantic:{row["semantic_score"]:.1f}  Skill:{row["skill_score"]:.1f}  '
            f'Exp:{row["experience_score"]:.1f}  Edu:{row["education_score"]:.1f}',
            f'  Experience: {row["experience_display"]}',
            f'  Education : {row["education_display"]}',
            f'  ✅ Matched : {row["skills_found"]}',
            f'  ❌ Missing : {row["skills_missing"]}',
            '',
        ]
    if len(df) > 0:
        summary += ['=' * 62, '🥇 TOP CANDIDATE REPORT:', '=' * 62,
                    df.iloc[0]['feedback_report']]
    return display_df, '\n'.join(summary)


def jobseeker_analyze_ui(pdf_file, job_description,
                          required_skills_str, nice_skills_str):
    if pdf_file is None:
        return 'Please upload your resume.', ''
    if not job_description.strip():
        return 'Please paste the job description.', ''
    if not required_skills_str.strip():
        return 'Please enter at least one required skill.', ''

    path            = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
    parsed          = parser.parse(path)
    required_skills = [s.strip().lower() for s in required_skills_str.split(',') if s.strip()]
    nice_skills     = [s.strip().lower() for s in nice_skills_str.split(',') if s.strip()]

    profile_lines = [
        '─' * 55,
        f'📄 File            : {parsed["file"]}',
        f'👤 Name            : {parsed["name"]}',
        f'📧 Email           : {parsed["email"]}',
        f'📱 Phone           : {parsed["phone"]}',
        f'🕐 Experience      : {parsed["experience_display"]}',
        f'🎓 Education       : {parsed["education_display"]}',
        '',
        f'🛠️  Skills ({len(parsed["skills"])}):',
        f'   {", ".join(parsed["skills"]) if parsed["skills"] else "None detected"}',
        f'🔧 Parse Engine    : {parsed.get("parse_engine", "unknown")}',
        f'📊 Parse Confidence: {parsed.get("parse_confidence", 0)}/100',
        '─' * 55,
    ]
    if parsed.get('parse_confidence', 100) < 50:
        profile_lines += [
            '⚠️  LOW PARSE CONFIDENCE — results may be unreliable.',
            '   Try a cleaner PDF or ensure OCR is installed.',
            '',
        ]
    analysis = matcher.analyze_for_jobseeker(
        parsed, job_description, required_skills, nice_skills)
    return '\n'.join(profile_lines), analysis


def get_audit_report():
    report      = parser.auditor.batch_report()
    field_rates = parser.auditor.get_field_failure_rates()
    lines       = [report, '']
    if field_rates:
        lines.append('📉 FIELD-LEVEL FAILURE RATES:')
        for field, rate in sorted(field_rates.items(), key=lambda x: -x[1]):
            bar = '█' * int(rate / 5)
            lines.append(f'  {field:15s}: {rate:5.1f}%  {bar}')
    else:
        lines.append('(No resumes parsed yet — use the other tabs first)')
    return '\n'.join(lines)


# ── Build UI ──────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(),
                   title='AI Resume Ranking App') as demo:

        gr.Markdown("""
        # 🤖 AI-Powered Resume Ranking App
        ### Intelligent Recruitment Automation — Production-Grade ML System
        *Fine-tuned SBERT · Hybrid Scoring · Explainable AI · pdfplumber · OCR*
        ---
        """)

        with gr.Tabs():

            # ── TAB 1: Resume Analysis ────────────────────────
            with gr.Tab('📄 Resume Analysis'):
                gr.Markdown("""
                Upload resumes to extract profiles.
                Multi-engine: **pdfplumber → PyPDF2 → OCR**
                Each resume gets a parse confidence score (0–100).
                """)
                resume_upload = gr.File(
                    label='Upload Resume(s) — PDF only',
                    file_types=['.pdf'], file_count='multiple')
                parse_btn = gr.Button('🔍 Parse Resumes', variant='primary')
                with gr.Row():
                    parse_output = gr.Textbox(
                        label='Extracted Profiles + Audit',
                        lines=30, max_lines=60)
                    json_out = gr.Code(
                        label='Structured JSON Output',
                        language='json', lines=25)
                parse_btn.click(fn=parse_resume_ui,
                                inputs=[resume_upload],
                                outputs=[parse_output, json_out])

            # ── TAB 2: Job Matching ───────────────────────────
            with gr.Tab('🎯 Job Matching & Ranking'):
                gr.Markdown('### Rank candidates against a job description')
                with gr.Row():
                    with gr.Column(scale=1):
                        match_upload  = gr.File(
                            label='Upload Resumes (PDF)',
                            file_types=['.pdf'], file_count='multiple')
                        job_title_inp = gr.Textbox(
                            label='Job Title',
                            value='Machine Learning Engineer')
                        required_inp  = gr.Textbox(
                            label='Required Skills (comma-separated)',
                            value=SAMPLE_REQUIRED_SKILLS, lines=3)
                        nice_inp      = gr.Textbox(
                            label='Nice-to-Have Skills (comma-separated)',
                            value=SAMPLE_NICE_SKILLS, lines=2)
                        with gr.Row():
                            min_exp_inp = gr.Slider(
                                label='Min Experience (years)',
                                minimum=0, maximum=20, value=3, step=0.5)
                            max_exp_inp = gr.Slider(
                                label='Max Experience (years)',
                                minimum=0, maximum=25, value=8, step=0.5)
                    with gr.Column(scale=1):
                        jd_inp = gr.Textbox(
                            label='Job Description',
                            value=SAMPLE_JD, lines=22)
                match_btn     = gr.Button('🚀 Rank Candidates',
                                          variant='primary', size='lg')
                ranking_table = gr.Dataframe(
                    label='🏆 Ranking Table', wrap=True)
                match_summary = gr.Textbox(
                    label='Detailed Analysis', lines=30, max_lines=60)
                match_btn.click(
                    fn=match_resumes_ui,
                    inputs=[match_upload, job_title_inp, jd_inp,
                            required_inp, nice_inp, min_exp_inp, max_exp_inp],
                    outputs=[ranking_table, match_summary])

            # ── TAB 3: Job-Seeker Analysis ────────────────────
            with gr.Tab('👤 Job-Seeker Analysis'):
                gr.Markdown("""
                ### How well does your resume match this job?
                Get a personalised fit score and upskilling suggestions.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        js_upload   = gr.File(
                            label='Upload Your Resume (PDF)',
                            file_types=['.pdf'], file_count='single')
                        js_req_inp  = gr.Textbox(
                            label='Required Skills (comma-separated)',
                            value=SAMPLE_REQUIRED_SKILLS, lines=3)
                        js_nice_inp = gr.Textbox(
                            label='Nice-to-Have Skills (comma-separated)',
                            value=SAMPLE_NICE_SKILLS, lines=2)
                    with gr.Column(scale=1):
                        js_jd_inp = gr.Textbox(
                            label='Job Description',
                            value=SAMPLE_JD, lines=20)
                js_btn      = gr.Button('🔍 Analyse My Resume',
                                        variant='primary', size='lg')
                js_profile  = gr.Textbox(label='Your Profile',
                                          lines=14, max_lines=20)
                js_analysis = gr.Textbox(label='Fit Analysis & Suggestions',
                                          lines=25, max_lines=50)
                js_btn.click(
                    fn=jobseeker_analyze_ui,
                    inputs=[js_upload, js_jd_inp, js_req_inp, js_nice_inp],
                    outputs=[js_profile, js_analysis])

            # ── TAB 4: Parse Quality Audit ────────────────────
            with gr.Tab('🔍 Parse Quality Audit'):
                gr.Markdown("""
                ## Parse Quality Audit
                Parse resumes in other tabs first, then refresh here.

                | Points | Field |
                |--------|-------|
                | +25 | Name |
                | +20 | Email |
                | +15 | Phone |
                | +20 | Skills (≥5 full, ≥2 half) |
                | +10 | Experience |
                | +10 | Education |

                Scores below 50 are flagged as low-confidence.
                """)
                audit_btn    = gr.Button('🔄 Refresh Audit', variant='secondary')
                audit_output = gr.Textbox(label='Audit Report',
                                           lines=25, max_lines=50)
                audit_btn.click(fn=get_audit_report,
                                inputs=[], outputs=[audit_output])

            # ── TAB 5: Model Status ───────────────────────────
            with gr.Tab('🤖 Model Status'):
                gr.Markdown(f"""
                ## Active Model
                **{embeddings_service.get_model_info()}**

                ---

                ## Scoring Weights

                | Factor | Weight |
                |--------|--------|
                | Semantic (SBERT) | 40% |
                | Skill Match | 35% |
                | Experience | 20% |
                | Education | 5% |

                ## Recommendation Tiers

                | Score | Label |
                |-------|-------|
                | ≥ 80 | 🟢 Excellent |
                | 60–79 | 🟡 Good |
                | 40–59 | 🟠 Fair |
                | < 40 | 🔴 Poor |
                """)

            # ── TAB 6: About ──────────────────────────────────
            with gr.Tab('ℹ️ About'):
                gr.Markdown("""
                ## 📖 User Guide

                **Resume Analysis** — Upload PDFs, extract profiles + parse audit.

                **Job Matching** — Upload multiple resumes + JD, get ranked table.

                **Job-Seeker Analysis** — Upload your own resume, get personal fit score.

                **Parse Quality Audit** — Confidence scores + field failure rates.

                ---

                ## 🔧 OCR on HuggingFace Spaces
                Add a `packages.txt` file containing:
```
                tesseract-ocr
                poppler-utils
```

                ---

                ## 🚀 Future Work
                - RAG-powered resume querying
                - ATS compatibility scoring
                - Bias & fairness detection
                - Skill-gap dashboards
                """)

    return demo


# ── Entry point ───────────────────────────────────────────────

demo = build_ui()
demo.launch()