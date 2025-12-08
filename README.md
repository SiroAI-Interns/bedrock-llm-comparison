# Bedrock LLM Comparison Platform

Enterprise-grade multi-model LLM comparison and analysis platform supporting AWS Bedrock, OpenAI, and Google Gemini.

## 🎯 Overview

A production-ready Python application for comparing responses across multiple Large Language Models (LLMs) with automated reporting, batch processing, and database integration capabilities.

### Key Features

- ✅ **Multi-Provider Support**: AWS Bedrock (Titan, LLaMA, Claude 3.5), OpenAI GPT-4o-mini, Google Gemini
- ✅ **Enterprise Architecture**: Modular, scalable, and maintainable codebase
- ✅ **Automated Reporting**: Excel exports with 5 comprehensive analysis sheets
- ✅ **Batch Processing**: Handle multiple prompts with parallel execution
- ✅ **Database Integration**: PostgreSQL/Supabase support for prompt management
- ✅ **PDF Protocol Processing**: Extract and cache protocol text (future-ready)
- ✅ **Configuration Management**: Environment-based settings with YAML configs
- ✅ **Production Logging**: Structured logging with rotation
- ✅ **CI/CD Ready**: GitHub Actions workflows included

## 📋 Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- OpenAI API Key (optional)
- Google Gemini API Key (optional)
- PostgreSQL database (optional, for prompt management)

## 🚀 Quick Start

### 1. Clone Repository

- git clone https://github.com/SiroAI-Interns/bedrock-llm-comparison.git
- cd bedrock-llm-comparison


### 2. Environment Setup

Create virtual environment
- python3 -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies
- make install
Or manually:
- pip install -r requirements.txt 


### 3. Configuration

Copy environment template
- cp .env.example .env
Edit .env with your credentials


### 4. Run Example

Single prompt test
- make run
Or directly:
- python scripts/run_single_prompt.py


## 📁 Project Structure



## 🔧 Configuration

### Environment Variables

Key environment variables (see `.env.example`):


### Model Configuration

Edit `config/models.yaml` to:
- Enable/disable specific models
- Adjust default parameters
- Configure regions

## 📊 Usage Examples

### Single Prompt Comparison

from app.services.comparison_service import ComparisonService
service = ComparisonService()results = service.compare_single_prompt(prompt=“Explain machine learning in 5 bullet points”,max_tokens=256,temperature=0.7)