# Team 30: RFP Analysis Tool

An automated analysis tool designed to help government contractors evaluate Request for Proposals (RFPs). The tool processes RFP PDFs, builds a vector index for text retrieval, and performs several analyses including eligibility checks, scoring assessments, risk evaluations, and more. It also includes an interactive chatbot agent in the clarification questions section to facilitate further Q&A about the RFP or the tool's functionalities.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)

## Overview

The **ConsultAdd RFP Analysis Tool** is built with Python and Streamlit, providing a user-friendly web interface to automate the analysis of government RFP documents. The tool extracts text from uploaded PDF files, creates a vector store for similarity search, and then performs a series of analyses:
- **Eligibility Analysis:** Determines if a company meets mandatory RFP requirements.
- **Scoring Assessment:** Evaluates proposal scoring criteria.
- **Risk Analysis:** Identifies and assesses risks associated with the RFP.
- **Clarification Questions:** Generates a list of professional questions to seek further clarification from the issuing agency.
- **Submission Checklist:** Produces a checklist for required submission documents.
- **Executive Summary:** Compiles a comprehensive summary of all analysis.
- **Interactive Chatbot:** Allows users to ask follow-up questions via an integrated chatbot interface.

## Features

- **PDF Processing:** Extracts text from uploaded PDF files using PyMuPDF.
- **Text Splitting & Vector Store:** Utilizes LangChain's RecursiveCharacterTextSplitter and FAISS with HuggingFace embeddings for efficient text retrieval.
- **Natural Language Processing:** Powered by a Groq-based ChatGroq model to analyze RFP content.
- **User Interface:** Built with Streamlit for an interactive web experience.
- **Interactive Q&A Chatbot:** Enables users to ask additional questions about the analysis or tool functionality.
- **Modular Analysis Functions:** Separate functions handle eligibility checks, scoring evaluation, risk analysis, and more.

## Architecture

The tool is structured into several key components:

1. **Environment Setup & Imports:**  
   - Loads necessary libraries and environment variables.
   - Configures logging and initializes Streamlit session state.

2. **Helper Functions:**  
   - **PDF Processing:** Extract text using PyMuPDF.
   - **Text Splitting:** Split text into chunks using LangChain.
   - **Vector Store Creation:** Build a FAISS vector store using HuggingFace embeddings.
   - **Model Invocation:** Send prompts to the ChatGroq model and retrieve responses.
   - **Markdown Table Parsing:** Convert markdown tables into Pandas DataFrames.

3. **Analysis Functions:**  
   - **Eligibility Analysis:** Checks if the company meets mandatory RFP requirements.
   - **Scoring Assessment:** Evaluates proposal scoring based on RFP criteria.
   - **Risk Analysis:** Identifies and assesses risks in the RFP.
   - **Clarification Questions:** Generates questions for the government agency.
   - **Submission Checklist:** Creates a comprehensive checklist for proposal submission.
   - **Executive Summary:** Compiles an overall summary of the RFP analysis.

4. **Chatbot Agent:**  
   - Provides an interactive Q&A experience in the Clarification Questions tab.

5. **User Interface:**  
   - Built with Streamlit, featuring file uploads, multiple analysis buttons, and tabbed results for easy navigation.
   - Uses session state to maintain data (analysis results, chat history, etc.) across user interactions.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/consultadd-rfp-analysis-tool.git
cd consultadd-rfp-analysis-tool
