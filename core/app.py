import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings  # HuggingFace embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
import tempfile
import json

from dotenv import load_dotenv
load_dotenv()

# Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

st.title("ATS Resume Score Checker")
st.markdown("Analyze your resume against ATS systems and get actionable insights")

# Initialize ATS Guidelines Vector Store (only once)
if "vectors" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create ATS best practices documents
    ats_guidelines = [
        """ATS Keyword Optimization: 
        - Include industry-specific keywords and technical skills
        - Use standard job titles aligned with the position
        - Include certifications and relevant technical terminology
        - Avoid generic terms, be specific about technologies used
        - Mirror keywords from job description
        - Use proper technical abbreviations (AWS, ML, Python, etc.)""",
        
        """ATS Formatting Best Practices:
        - Use simple, clean formatting without tables, columns, or graphics
        - Avoid images, logos, headers, and footers that break text flow
        - Use standard fonts (Arial, Calibri, Times New Roman)
        - Stick to .pdf or .docx format for compatibility
        - Keep margins between 0.5 and 1 inch
        - Use single spacing between lines
        - Use standard bullet points, not special characters
        - Avoid headers and footers that may be skipped by ATS""",
        
        """ATS Content Structure:
        - Clear section headings: Contact, Summary, Experience, Skills, Education
        - Contact information at the top with name, phone, email, city
        - Professional summary (2-3 lines, achievement-focused)
        - Work experience with company, job title, dates in reverse chronological order
        - Use action verbs: developed, implemented, designed, led, created
        - Quantifiable achievements with metrics (increased by X%, reduced by Y)
        - Skills section with technical and soft skills
        - Education with degree, school, graduation date
        - Optional: certifications, projects, languages""",
        
        """Resume Length and Readability:
        - Keep resume to 1 page for entry-level, 2 pages for experienced professionals
        - Use clear hierarchy with bold headings and consistent formatting
        - Keep sentences concise and action-oriented
        - Avoid lengthy paragraphs, use bullet points
        - Use proper grammar and spelling (critical for ATS)
        - Ensure consistent date formatting throughout
        - Remove unnecessary details and focus on achievements
        - White space is important for readability""",
        
        """Skills Section Optimization:
        - Create a dedicated technical skills section
        - List programming languages, frameworks, tools, and platforms
        - Organize by category: Languages, Frameworks, Tools, Databases, Platforms
        - Include both hard and soft skills
        - Use industry-standard terminology
        - Match skills from job description
        - Order skills by relevance to target role
        - Use commas or bullet points consistently""",
        
        """Experience Section Best Practices:
        - Start with most recent job (reverse chronological order)
        - Include company name, job title, dates (Month Year - Month Year)
        - Use 3-5 bullet points per job (more for recent/relevant roles)
        - Focus on achievements and impact, not just duties
        - Include metrics and quantifiable results
        - Use action verbs to start each bullet
        - Show progression and increasing responsibility
        - Highlight relevant experience for target role""",
        
        """ATS Compliance Checklist:
        - File format is PDF or DOCX
        - No images, graphics, or visual elements
        - No tables or multi-column layouts
        - Standard fonts only
        - Proper spacing and margins
        - No headers or footers
        - Standard bullet points (-, •, or *)
        - No special characters or symbols
        - Clear section headings
        - Contact information at top""",
        
        """Red Flags that Damage ATS Score:
        - Unusual formatting or creative designs
        - Images, logos, or QR codes
        - Colored text or unusual fonts
        - Tables or multi-column layouts
        - Excessive use of special characters
        - Inconsistent date formats
        - Gaps in employment without explanation
        - Generic or vague descriptions
        - Poor grammar or spelling errors
        - Irrelevant information or outdated work"""
    ]
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=100
    )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
        [type('Document', (), {'page_content': guideline, 'metadata': {}})() for guideline in ats_guidelines]
    )
    
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents, 
        st.session_state.embeddings
    )


# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key, 
    # model_name="llama-3.1-8b-instant"  # High accuracy model
    model_name="qwen/qwen3-32b"
)

# Scoring Prompt Template
prompt_template = ChatPromptTemplate.from_template(
    """
Analyze the following resume based on ATS (Applicant Tracking System) best practices and standards.
Provide a detailed ATS score from 0-100 and specific feedback.

ATS GUIDELINES CONTEXT:
{context}

RESUME CONTENT:
{input}

Please provide your analysis in the following JSON format ONLY (no additional text):
{{
    "overall_ats_score": <number 0-100>,
    "keyword_optimization": {{"score": <number>, "feedback": "<string>"}},
    "formatting_and_structure": {{"score": <number>, "feedback": "<string>"}},
    "content_quality": {{"score": <number>, "feedback": "<string>"}},
    "readability_and_clarity": {{"score": <number>, "feedback": "<string>"}},
    "compliance_and_best_practices": {{"score": <number>, "feedback": "<string>"}},
    "critical_issues": ["<issue1>", "<issue2>", ...],
    "improvement_recommendations": ["<recommendation1>", "<recommendation2>", ...],
    "strengths": ["<strength1>", "<strength2>", ...]
}}

Be precise, actionable, and focus on ATS compatibility. Do not include any text outside the JSON format.
"""
)

# Create the RAG chain
documents_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 6})
retrieval_chain = create_retrieval_chain(retriever, documents_chain)


# Streamlit UI
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload PDF or DOCX resume", type=["pdf", "docx"])

with col2:
    st.subheader("Target Role (Optional)")
    job_title = st.text_input("Job title you're targeting (helps improve matching)")


if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Load resume content
        st.info("Loading resume...")
        loader = PyPDFLoader(tmp_file_path)
        resume_docs = loader.load()
        resume_content = "\n".join([doc.page_content for doc in resume_docs])
        
        if len(resume_content.strip()) == 0:
            st.error("❌ Could not extract text from PDF. Please ensure the PDF contains selectable text.")
        else:
            # Add job title context if provided
            if job_title.strip():
                resume_content_with_context = f"TARGET ROLE: {job_title}\n\nRESUME:\n{resume_content}"
            else:
                resume_content_with_context = resume_content
            
            # Analyze resume
            if st.button("Analyze Resume", key="analyze_btn"):
                st.info("Analyzing your resume against ATS standards...")
                start = time.process_time()
                
                response = retrieval_chain.invoke({
                    "input": resume_content_with_context
                })
                
                analysis_time = time.process_time() - start
                
                st.success(f"✅ Analysis complete in {analysis_time:.2f} seconds")
                
                # Parse the response
                try:
                    # Extract JSON from response
                    answer_text = response['answer'].strip()
                    
                    # Try to parse JSON
                    try:
                        score_data = json.loads(answer_text)
                    except json.JSONDecodeError:
                        # If direct parsing fails, try to extract JSON from text
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', answer_text)
                        if json_match:
                            score_data = json.loads(json_match.group())
                        else:
                            st.error("⚠️ Could not parse ATS analysis. Please try again.")
                            score_data = None
                    
                    if score_data:
                        # Display Overall Score
                        overall_score = score_data.get("overall_ats_score", 0)
                        st.markdown("---")
                        
                        col_score, col_gauge = st.columns([2, 1])
                        with col_score:
                            st.markdown(f"## 📊 Overall ATS Score: **{overall_score}/100**")
                        
                        with col_gauge:
                            if overall_score >= 80:
                                st.success("🟢 Excellent")
                            elif overall_score >= 60:
                                st.warning("🟡 Good")
                            else:
                                st.error("🔴 Needs Improvement")
                        
                        # Category Breakdown
                        st.markdown("---")
                        st.subheader("Category Breakdown")
                        
                        categories = {
                            "keyword_optimization": "Keyword Optimization",
                            "formatting_and_structure": "Formatting & Structure",
                            "content_quality": "Content Quality",
                            "readability_and_clarity": "Readability & Clarity",
                            "compliance_and_best_practices": "Compliance & Best Practices"
                        }
                        
                        cols = st.columns(5)
                        for idx, (key, title) in enumerate(categories.items()):
                            with cols[idx]:
                                if key in score_data:
                                    category_score = score_data[key].get("score", 0)
                                    st.metric(
                                        label=title,
                                        value=f"{category_score}/100"
                                    )
                        
                        # Detailed Feedback
                        st.markdown("---")
                        st.subheader("Detailed Feedback")
                        
                        for key, title in categories.items():
                            if key in score_data:
                                with st.expander(f"{title} (Score: {score_data[key].get('score', 0)}/100)"):
                                    st.write(score_data[key].get("feedback", "No feedback available"))
                        
                        # Strengths
                        if "strengths" in score_data and score_data["strengths"]:
                            st.markdown("---")
                            st.subheader("Strengths")
                            for strength in score_data["strengths"]:
                                st.success(f"✓ {strength}")
                        
                        # Critical Issues
                        if "critical_issues" in score_data and score_data["critical_issues"]:
                            st.markdown("---")
                            st.subheader("Critical Issues")
                            for issue in score_data["critical_issues"]:
                                st.error(f"✗ {issue}")
                        
                        # Recommendations
                        if "improvement_recommendations" in score_data and score_data["improvement_recommendations"]:
                            st.markdown("---")
                            st.subheader("Improvement Recommendations")
                            for idx, rec in enumerate(score_data["improvement_recommendations"], 1):
                                st.info(f"{idx}. {rec}")
                
                except Exception as e:
                    st.error(f"Error parsing analysis: {str(e)}")
                
                # Document Similarity Search (showing context used)
                with st.expander("ATS Guidelines Context Used"):
                    st.markdown("**Top matching ATS guidelines retrieved for analysis:**")
                    for i, doc in enumerate(response["context"][:3], 1):
                        st.markdown(f"**Guideline {i}:**")
                        st.write(doc.page_content)
                        st.write("---")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

else:
    st.info("Please upload a resume to get started with ATS analysis")