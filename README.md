# HR Agent – Top Candidate Selection

An LLM agent that ranks resumes against a job description and picks the top candidates.

## How it works
1. **Job description → JSON**: the plain-text JD is converted into a structured `JobDescription` (skills, experience, education, responsibilities, ...).
2. **Resumes → JSON**: every `.pdf` / `.docx` in the resume folder is parsed into a structured `Resume`.
3. **Evaluation**: each resume is scored against the JD (0–100) on skills, experience, education and projects/certifications, with matched/missing skills, strengths, gaps and reasoning.
4. **Suitability %**: weighted average computed in Python (skills 40%, experience 30%, education 15%, projects/certs 15%).
5. **Top candidates**: the top N are sent back to the LLM to explain why they were selected and list the skills that made them stand out.

All LLM outputs are forced into JSON schemas and validated with pydantic.

## Setup
```bash
pip install -r requirements.txt
```
Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

## Usage
```bash
python hr_agent.py                                   # uses job_description.txt and ./resume
python hr_agent.py --jd my_jd.txt --resumes path/to/resumes --out output --top 2
```

## Output (`output/`)
- `job_description.json` – structured JD
- `parsed_resumes/*.json` – one structured JSON per resume
- `final_report.json` – top candidates, comparison summary and all candidates ranked
