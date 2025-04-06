import os
import logging
import fitz  # PyMuPDF
import json
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Initialize session state
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "full_rfp_text" not in st.session_state:
    st.session_state.full_rfp_text = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "is_eligible" not in st.session_state:
    st.session_state.is_eligible = None
if "eligibility_details" not in st.session_state:
    st.session_state.eligibility_details = None

# --- Configuration ---
# Get API key from either st.secrets or environment variables
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
MAX_CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

# Default ConsultAdd company profile
COMPANY_PROFILE = {
    "legal_name": "ConsultAdd, Inc.",
    "address": "100 Technology Drive, Suite 300, San Francisco, CA 94105",
    "employees": 450,
    "annual_revenue": "$65M",
    "years_in_business": 8,
    "certifications": ["ISO 9001", "CMMI Level 3", "SOC 2 Type II"],
    "industry_sectors": ["Government IT Services", "Healthcare Solutions", "Financial Services Technology"],
    "past_performance": [
        "CDC Data Management System (2022-Present)", 
        "Department of Labor IT Staffing (2020-2023)", 
        "State of California Enterprise Resource Planning (2021-Present)"
    ],
    "capabilities": ["IT Consulting", "Staff Augmentation", "System Integration", "Custom Software Development"],
    "state_registrations": ["California", "Texas", "Virginia", "New York", "Florida"],
    "small_business_status": False,
    "socioeconomic_status": ["Large Business"],
    "duns_number": "123456789",
    "cage_code": "AB123",
    "naics_codes": ["541512", "541511", "541519"]
}

# --- Helper Functions ---
def initialize_groq():
    """Initialize Groq Chat model"""
    if not GROQ_API_KEY:
        st.error("Groq API key not found!")
        return None
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0.2
    )

def load_pdf_text(uploaded_file):
    """Extract text from uploaded PDF"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        doc = fitz.open(tmp_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        # Make sure to close the document before deletion
        doc.close()
        
        try:
            os.unlink(tmp_path)
        except Exception as e:
            logging.warning(f"Could not delete temporary file {tmp_path}: {str(e)}")
        
        return full_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""

def split_text_into_chunks(text, chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks for processing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text):
    """Create a FAISS vector store from the text using HuggingFace embeddings"""
    chunks = split_text_into_chunks(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def retrieve_relevant_chunks(query, vector_store, k=4):
    """Retrieve the top k relevant chunks from the vector store for a given query"""
    results = vector_store.similarity_search(query, k=k)
    # Combine the retrieved documents into one string for context
    combined_text = "\n\n".join([doc.page_content for doc in results])
    return combined_text

def analyze_with_groq(prompt: str) -> str:
    """Execute analysis using Groq AI"""
    try:
        llm = initialize_groq()
        if not llm:
            return "Error: API key not configured"
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logging.error(f"Groq API error: {str(e)}")
        return f"Error: {str(e)}"

def parse_markdown_table(text: str) -> pd.DataFrame:
    """
    Parse a markdown table from a given text into a Pandas DataFrame.
    Expects the table rows to start with a pipe '|' character.
    """
    # Extract lines that look like table rows
    lines = [line for line in text.split("\n") if line.strip().startswith("|")]
    if len(lines) < 3:
        return pd.DataFrame()  # No valid table found
    header_line = lines[0]
    separator_line = lines[1]
    data_lines = lines[2:]
    
    # Split header by pipe and remove empty strings
    headers = [h.strip() for h in header_line.split("|") if h.strip()]
    
    # Parse each row
    rows = []
    for line in data_lines:
        # Skip separator lines
        if set(line.strip()) <= set("-| "):
            continue
        row = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(row) == len(headers):
            rows.append(row)
    
    df = pd.DataFrame(rows, columns=headers)
    return df

# --- Chatbot Agent Function ---
def chatbot_agent(question: str) -> str:
    """
    Chatbot agent to answer additional questions about the RFP analysis tool,
    the bot's capabilities, or related topics in a clear and professional manner.
    """
    llm = initialize_groq()
    if not llm:
        return "Error: Chatbot agent not available due to missing API key."
    
    prompt = f"""
    You are a chatbot assistant specialized in government RFP analysis and proposal evaluation. 
    Answer the following question clearly and professionally:
    
    Question: {question}
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- Analysis Functions ---
def check_eligibility(rfp_text: str, company_profile: dict) -> tuple:
    """
    First check if ConsultAdd is legally eligible to bid on the RFP.
    Returns (is_eligible, detailed_explanation)
    """
    # Retrieve relevant sections for eligibility criteria
    query = "eligibility criteria requirements mandatory qualifications certifications past performance mandatory bidder"
    relevant_context = retrieve_relevant_chunks(query, st.session_state.vector_store, k=5)
    
    prompt = f"""
    You are an expert in government RFP compliance analysis. Your task is to determine if ConsultAdd is ELIGIBLE to bid on this RFP.
    
    First, carefully analyze the RFP context below to identify ALL mandatory requirements:
    
    RFP CONTEXT:
    {relevant_context}
    
    COMPANY PROFILE:
    {json.dumps(company_profile, indent=2)}
    
    Your analysis must be thorough and structured as follows:
    
    1. ELIGIBILITY DETERMINATION:
       Start with a clear "ELIGIBLE" or "NOT ELIGIBLE" verdict
    
    2. MANDATORY REQUIREMENTS:
       List all mandatory qualification requirements found in the RFP
    
    3. REQUIREMENT MATCHING:
       For each requirement, indicate whether ConsultAdd meets it with evidence from the company profile
    
    4. DEAL-BREAKERS:
       Identify any absolute deal-breakers that would disqualify ConsultAdd
    
    5. ELIGIBILITY SUMMARY:
       Provide a concise justification for your eligibility determination
    
    Format your response using markdown with clear sections and bullet points.
    """
    
    response = analyze_with_groq(prompt)
    
    # Correct eligibility check:
    # If "NOT ELIGIBLE" is present, then it's not eligible.
    # Otherwise, if "ELIGIBLE" is present, then it is eligible.
    if "NOT ELIGIBLE" in response[:500]:
        is_eligible = False
    elif "ELIGIBLE" in response[:500]:
        is_eligible = True
    else:
        is_eligible = False  # Default to not eligible if unclear
    
    return is_eligible, response

def evaluate_rfp_scoring(rfp_text: str, company_profile: dict) -> str:
    """Evaluate and score the RFP based on multiple criteria"""
    query = "evaluation criteria scoring points weights proposal rating scale methodology"
    relevant_context = retrieve_relevant_chunks(query, st.session_state.vector_store, k=4)
    
    prompt = f"""
    You are an expert government proposal evaluator. Analyze the RFP and score ConsultAdd's potential proposal based on the evaluation criteria.
    
    RFP CONTEXT:
    {relevant_context}
    
    COMPANY PROFILE:
    {json.dumps(company_profile, indent=2)}
    
    Create a comprehensive scoring assessment with these sections:
    
    1. EVALUATION METHODOLOGY:
       Extract the scoring methodology from the RFP (point system, weights, etc.)
    
    2. SCORING BREAKDOWN:
       Create a detailed table with:
       - Each evaluation criterion 
       - Weight/points possible
       - Projected score for ConsultAdd
       - Brief justification
    
    3. OVERALL SCORE:
       Calculate ConsultAdd's projected total score and percentage
    
    4. COMPETITIVE POSITIONING:
       Assess how competitive this proposal would be
    
    5. SCORE IMPROVEMENT RECOMMENDATIONS:
       Specific actions to improve scoring potential
    
    Format as a professional report with tables and clear sections using markdown.
    """
    
    return analyze_with_groq(prompt)

def analyze_risk_factors(rfp_text: str) -> str:
    """Perform detailed risk analysis on the RFP"""
    query = "risk liability payment terms insurance contract requirements termination deadlines penalties confidentiality ip rights"
    relevant_context = retrieve_relevant_chunks(query, st.session_state.vector_store, k=4)
    
    prompt = f"""
    You are an expert risk analyst specializing in government contracts. Perform a comprehensive risk analysis of this RFP.
    
    RFP CONTEXT:
    {relevant_context}
    
    Create a detailed risk assessment with these sections:
    
    1. RISK FACTOR MATRIX:
       Create a table with these columns:
       - Risk Category
       - Risk Description
       - Probability (High/Medium/Low)
       - Impact (High/Medium/Low)
       - Risk Score (1-25)
       - Mitigation Strategy
    
    2. HIGH-PRIORITY RISKS:
       Identify the top 3-5 most critical risks that require immediate attention
    
    3. CONTRACTUAL RED FLAGS:
       Identify concerning clauses or terms that could disadvantage ConsultAdd
    
    4. RISK MITIGATION PLAN:
       Propose specific strategies to address the highest risks
    
    5. GO/NO-GO RISK ASSESSMENT:
       Provide an overall risk rating that could inform the bid decision
    
    Format as a professional report with clear sections using markdown.
    """
    
    return analyze_with_groq(prompt)

def generate_clarification_questions(rfp_text: str) -> str:
    """Generate relevant questions to ask the government agency about the RFP"""
    query = "unclear requirements ambiguity specifications scope deliverables timeline budget responsibility"
    relevant_context = retrieve_relevant_chunks(query, st.session_state.vector_store, k=4)
    
    prompt = f"""
    You are an expert government proposal manager. Identify areas in the RFP that need clarification and create a list of professional questions to ask the government agency.
    
    RFP CONTEXT:
    {relevant_context}
    
    Generate a comprehensive set of clarification questions with these categories:
    
    1. SCOPE & REQUIREMENTS QUESTIONS:
       Questions about unclear deliverables, specifications, or scope boundaries
    
    2. TECHNICAL QUESTIONS:
       Questions about technical specifications, integration points, or methodologies
    
    3. CONTRACTUAL QUESTIONS:
       Questions about terms, conditions, or legal requirements
    
    4. EVALUATION QUESTIONS:
       Questions about how proposals will be evaluated or scored
    
    5. TIMELINE & LOGISTICS QUESTIONS:
       Questions about deadlines, delivery schedules, or logistical requirements
    
    Format each question professionally as you would submit to a government agency. For each question:
    - Reference the specific RFP section/page when possible
    - Explain why the clarification is needed
    - Suggest a potential answer if appropriate
    
    Format using markdown with clear sections and numbered questions.
    """
    
    return analyze_with_groq(prompt)

def generate_submission_checklist(rfp_text: str) -> str:
    """Generate a detailed submission checklist based on RFP requirements"""
    query = "submission requirements documents format deadline instructions proposal organization"
    relevant_context = retrieve_relevant_chunks(query, st.session_state.vector_store, k=4)
    
    prompt = f"""
    You are an expert proposal manager. Create a comprehensive submission checklist for this RFP.
    specifically mention dates , deadlines and related .
    
    RFP CONTEXT:
    {relevant_context}
    
    Generate a detailed submission checklist with these sections:
    
    1. KEY DATES & DEADLINES:
       List all important dates in chronological order
    
    2. SUBMISSION LOGISTICS:
       - Method of submission (electronic/physical)
       - Number of copies
       - Formatting requirements
       - Page limits
    
    3. REQUIRED DOCUMENTS CHECKLIST:
       Create a comprehensive list of all required documents, forms, and attachments
    
    4. PROPOSAL STRUCTURE:
       Outline the required organization and sections of the proposal
    
    5. CERTIFICATION REQUIREMENTS:
       List all certifications and statements that must be included
    
    6. COMMON SUBMISSION PITFALLS:
       Identify common mistakes to avoid
    
    Format as a practical, actionable checklist using markdown with checkboxes and clear sections.
    """
    
    return analyze_with_groq(prompt)

def generate_executive_summary(rfp_text: str, company_profile: dict, is_eligible: bool, 
                              eligibility_details: str = None, scoring: str = None, 
                              risks: str = None) -> str:
    """Generate an executive summary of the RFP analysis"""
    # Get basic RFP info
    query = "rfp title scope objectives project summary agency department"
    relevant_context = retrieve_relevant_chunks(query, st.session_state.vector_store, k=3)
    
    # Prepare input data with conditionals for post-eligibility analysis
    prompt_data = {
        "rfp_context": relevant_context,
        "company_profile": json.dumps(company_profile, indent=2),
        "is_eligible": "Yes" if is_eligible else "No"
    }
    
    # Build the prompt conditionally
    prompt = f"""
    You are a senior proposal consultant. Create a concise executive summary of this RFP analysis.
    
    RFP CONTEXT:
    {prompt_data['rfp_context']}
    
    COMPANY PROFILE:
    {prompt_data['company_profile']}
    
    ELIGIBILITY DETERMINATION:
    {prompt_data['is_eligible']}
    
    """
    
    # Add additional sections only if eligible and data exists
    if is_eligible and eligibility_details:
        prompt += f"ELIGIBILITY DETAILS:\n{eligibility_details[:1000]}...\n\n"
    
    if is_eligible and scoring:
        prompt += f"SCORING ASSESSMENT:\n{scoring[:1000]}...\n\n"
    
    if is_eligible and risks:
        prompt += f"RISK ANALYSIS:\n{risks[:1000]}...\n\n"
    
    prompt += """
    Create a concise executive summary with these sections:
    
    1. RFP OVERVIEW:
       Brief description of the opportunity
    
    2. ELIGIBILITY ASSESSMENT:
       Clear statement on ConsultAdd's eligibility
    """
    
    # Add additional sections to output only if eligible
    if is_eligible:
        prompt += """
    3. COMPETITIVE POSITION:
       Assessment of ConsultAdd's competitive position
    
    4. KEY RISKS & CONSIDERATIONS:
       Summary of critical factors to consider
    
    5. GO/NO-GO RECOMMENDATION:
       Clear recommendation with brief rationale
    """
    
    prompt += """
    Format as a professional one-page executive summary using markdown with clear sections.
    """
    
    return analyze_with_groq(prompt)

# --- Main UI ---
def main():
    st.set_page_config(
        page_title="ConsultAdd RFP Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar Configuration
    st.sidebar.title("RFP Analysis Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload RFP PDF", type="pdf")
   
    # Company Profile Editor
    with st.sidebar.expander("Company Profile Editor", expanded=False):
        company_profile_json = st.text_area(
            "Edit ConsultAdd Profile (JSON)",
            value=json.dumps(COMPANY_PROFILE, indent=2),
            height=300
        )
        
        # Parse company profile with error handling
        try:
            company_profile = json.loads(company_profile_json)
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON: {str(e)}")
            company_profile = COMPANY_PROFILE
    
    # Main Interface
    st.title("ConsultAdd RFP Analysis Tool")
    st.markdown("### Automated RFP Analysis for Government Contracts")
   
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("📄 Process RFP Document", use_container_width=True):
                with st.spinner("Processing PDF and building vector index..."):
                    # Process PDF
                    rfp_text = load_pdf_text(uploaded_file)
                    
                    if not rfp_text:
                        st.error("Failed to extract text from PDF. Please check the file and try again.")
                    else:
                        st.session_state.full_rfp_text = rfp_text
                        st.success(f"RFP document processed! ({len(rfp_text)} characters extracted)")
                        # Create vector store for the RFP text
                        st.session_state.vector_store = create_vector_store(rfp_text)
                        st.info("Vector index created. Ready for eligibility check.")
        
        with col2:
            if st.session_state.full_rfp_text and st.session_state.vector_store:
                if st.button("🔍 Check Eligibility First", use_container_width=True):
                    # First check eligibility
                    with st.spinner("Analyzing eligibility requirements..."):
                        is_eligible, eligibility_details = check_eligibility(
                            st.session_state.full_rfp_text, company_profile
                        )
                        
                        # Store results in session state
                        st.session_state.is_eligible = is_eligible
                        st.session_state.eligibility_details = eligibility_details
                        st.session_state.analysis_results["eligibility"] = eligibility_details
                        
                        if is_eligible:
                            st.success("✅ ConsultAdd is ELIGIBLE to bid on this RFP!")
                        else:
                            st.error("❌ ConsultAdd is NOT ELIGIBLE to bid on this RFP.")
                
                # Only show further analysis options if eligible
                if "is_eligible" in st.session_state and st.session_state.is_eligible:
                    if st.button("📊 Perform Full Analysis", use_container_width=True):
                        with st.spinner("Performing comprehensive RFP analysis..."):
                            # Run all other analyses (sequentially here, but can be parallelized)
                            scoring_result = evaluate_rfp_scoring(
                                st.session_state.full_rfp_text, company_profile
                            )
                            
                            risk_result = analyze_risk_factors(st.session_state.full_rfp_text)
                            
                            questions_result = generate_clarification_questions(
                                st.session_state.full_rfp_text
                            )
                            
                            checklist_result = generate_submission_checklist(
                                st.session_state.full_rfp_text
                            )
                            
                            # Generate executive summary
                            executive_summary = generate_executive_summary(
                                st.session_state.full_rfp_text,
                                company_profile,
                                st.session_state.is_eligible,
                                st.session_state.eligibility_details,
                                scoring_result,
                                risk_result
                            )
                            
                            # Store all results
                            st.session_state.analysis_results.update({
                                "scoring": scoring_result,
                                "risks": risk_result,
                                "questions": questions_result,
                                "checklist": checklist_result,
                                "executive_summary": executive_summary
                            })
                            
                            st.success("✅ Complete RFP analysis finished!")
    
    # Display Results
    if "is_eligible" in st.session_state:
        st.markdown("## Eligibility Analysis")
        st.markdown(st.session_state.analysis_results.get("eligibility", "No eligibility analysis available."))
        
        if st.session_state.is_eligible and "scoring" in st.session_state.analysis_results:
            # Create tabs for different analysis sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Executive Summary", 
                "🔢 Scoring Assessment", 
                "⚠ Risk Analysis", 
                "❓ Clarification Questions",
                "✅ Submission Checklist"
            ])
            
            with tab1:
                st.markdown(st.session_state.analysis_results.get("executive_summary", "Executive summary not available."))
                
            with tab2:
                st.markdown("### Scoring Assessment")
                st.markdown(st.session_state.analysis_results.get("scoring", "Scoring assessment not available."))
                
                # Display scoring table and charts if available
                with st.expander("View Scoring Breakdown Table and Charts"):
                    scoring_text = st.session_state.analysis_results.get("scoring", "")
                    df = parse_markdown_table(scoring_text)
                    if not df.empty:
                        st.subheader("Scoring Breakdown Table")
                        st.dataframe(df)
                        # Convert numeric columns if possible
                        numeric_cols = ["Weight/Points Possible", "Projected Score for ConsultAdd"]
                        for col in numeric_cols:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Bar chart for Projected Score
                        if "Evaluation Criterion" in df.columns and "Projected Score for ConsultAdd" in df.columns:
                            st.subheader("Projected Score by Evaluation Criterion")
                            score_chart = df.set_index("Evaluation Criterion")["Projected Score for ConsultAdd"]
                            st.bar_chart(score_chart)
                        # Bar chart for Weight/Points Possible
                        if "Evaluation Criterion" in df.columns and "Weight/Points Possible" in df.columns:
                            st.subheader("Weight/Points Possible by Evaluation Criterion")
                            weight_chart = df.set_index("Evaluation Criterion")["Weight/Points Possible"]
                            st.bar_chart(weight_chart)
                    else:
                        st.info("No scoring table data available to display charts.")
                
            with tab3:
                st.markdown(st.session_state.analysis_results.get("risks", "Risk analysis not available."))
                
            with tab4:
                st.markdown(st.session_state.analysis_results.get("questions", "Clarification questions not available."))
                
                # --- Added Chatbot Agent for Further QnA ---
                st.markdown("### Ask Further Questions")
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                # Input for user question
                user_question = st.text_input("Enter your question", key="chat_question_input")
                if st.button("Submit Question", key="chat_submit_button") and user_question:
                    answer = chatbot_agent(user_question)
                    st.session_state.chat_history.append({"question": user_question, "answer": answer})
                
                # Display chat history
                if st.session_state.chat_history:
                    st.markdown("#### Chat History")
                    for chat in st.session_state.chat_history:
                        st.markdown(f"Q: {chat['question']}")
                        st.markdown(f"A: {chat['answer']}")
                # --- End Chatbot Agent Section ---
                
            with tab5:
                st.markdown(st.session_state.analysis_results.get("checklist", "Submission checklist not available."))
    
    # Optional: Allow user to view the extracted text
    if st.session_state.full_rfp_text:
        with st.expander("📄 View Extracted RFP Text", expanded=False):
            st.text_area("Extracted RFP Text", st.session_state.full_rfp_text, height=300)
    
    st.markdown("---")
    st.caption("ConsultAdd RFP Analysis Tool — Powered by LLaMA 3 via Groq API")

if _name_ == "_main_":
    main()
