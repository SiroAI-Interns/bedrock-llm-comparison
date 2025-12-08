# Bedrock LLM Comparison Platform

Multi-model LLM comparison and analysis platform supporting AWS Bedrock, OpenAI, and Google Gemini.

## Overview

A production-ready Python application for comparing responses across multiple Large Language Models (LLMs) with automated reporting, batch processing, and database integration capabilities.

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- OpenAI API Key (optional)
- Google Gemini API Key (optional)
- PostgreSQL database (optional, for prompt management)

## Quick Start

### 1. Clone Repository

git clone https://github.com/SIROAI-Interns/bedrock-llm-comparison.git
cd bedrock-llm-comparison

### 2. Environment Setup

Create virtual environment:
python3 -m venv venv
source venv/bin/activate

Install dependencies:
make install

Or manually:
pip install -r requirements.txt

### 3. Configuration

Copy environment template:
cp .env.example .env

Edit .env with your credentials using your preferred editor.

Required environment variables:
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

### 4. Run Example

Single prompt test:
make run

Or directly:
python scripts/run_single_prompt.py

## Project Structure

bedrock-llm-comparison/
├── app/                    # Application code
│   ├── core/              # LLM clients (Bedrock, OpenAI, Gemini)
│   ├── services/          # Business logic layer
│   ├── utils/             # Utilities and helpers
│   └── models/            # Data models
├── config/                # Configuration files
│   ├── settings.py        # Application settings
│   └── models.yaml        # Model configurations
├── data/                  # Data directory (gitignored)
│   ├── input/            # Input prompts and protocols
│   ├── output/           # Reports and logs
│   └── cache/            # PDF text cache
├── scripts/              # Execution scripts
│   ├── run_single_prompt.py
│   └── run_batch_comparison.py
├── tests/                # Test suite
│   ├── unit/
│   └── integration/
├── docs/                 # Documentation
├── docker/               # Docker configurations
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── Makefile             # Common commands
└── README.md

## Configuration

### Environment Variables

Key environment variables (see .env.example):

AWS Credentials:
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

Bedrock Regions:
BEDROCK_TITAN_REGION=us-east-1
BEDROCK_LLAMA_REGION=us-west-2
BEDROCK_CLAUDE_REGION=us-east-1

API Keys:
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key

Application Settings:
LOG_LEVEL=INFO
CACHE_ENABLED=true
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=512

### Model Configuration

Edit config/models.yaml to:
- Enable/disable specific models
- Adjust default parameters
- Configure regions

## Usage Examples

### Single Prompt Comparison

from app.services.comparison_service import ComparisonService

service = ComparisonService()
results = service.compare_single_prompt(
    prompt="Explain machine learning in 5 bullet points",
    max_tokens=256,
    temperature=0.7
)

### Batch Processing

from app.services.comparison_service import ComparisonService

service = ComparisonService()
prompts = [
    "Explain overfitting",
    "What is gradient descent?",
    "Describe neural networks"
]

service.compare_batch(prompts, output_file="batch_results.xlsx")

### From Text File

from app.services.comparison_service import ComparisonService

service = ComparisonService()

with open('data/input/prompts/prompts.txt', 'r') as f:
    prompts = [line.strip() for line in f if line.strip()]

service.compare_batch(prompts)

### Custom Settings Per Prompt

from app.services.comparison_service import ComparisonService

service = ComparisonService()

Medical query with conservative settings:
service.compare_single_prompt(
    prompt="Explain diabetes treatment options",
    temperature=0.3,
    max_tokens=200
)

Creative task with higher temperature:
service.compare_single_prompt(
    prompt="Generate marketing copy",
    temperature=1.0,
    max_tokens=500
)

## Excel Output

The tool generates Excel files with 5 comprehensive sheets:

1. Summary: Overall statistics including total responses, unique providers, and models
2. Provider_Model_Breakdown: Response counts per model for quick comparison
3. Responses_Summary: Compact view with timestamps, parameters, and metadata
4. Full_Responses: Complete response texts from all models
5. Model_Comparison: Statistical analysis including average, min, and max response lengths

Output files are saved to data/output/reports/ by default.


## Development

### Install Development Dependencies

pip install -r requirements-dev.txt

### Run Tests

Run all tests:
make test

Run with coverage:
pytest tests/ --cov=app --cov-report=html

Run specific test file:
pytest tests/unit/test_bedrock_client.py -v

### Code Quality

Format code:
make format

Lint code:
make lint

Run all quality checks:
make format && make lint && make test

## Docker Support

Build image:
docker-compose build

Run container:
docker-compose up

Run with custom environment file:
docker-compose --env-file .env.production up


## Troubleshooting

### AWS Credentials Issues

Verify AWS credentials:
aws sts get-caller-identity

Configure AWS CLI:
aws configure

### Import Errors

Ensure you're in the project root:
cd /path/to/bedrock-llm-comparison

Verify PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

### Gemini Empty Responses

Known issue: Gemini may return empty responses in some cases. This is under investigation. The model is disabled by default in config/models.yaml.


## Documentation

Detailed documentation available in docs/:

- Architecture Overview
- API Reference
- Deployment Guide
- User Guide

## License

MIT License - see LICENSE file for details.



- Initial release
- AWS Bedrock integration (Titan, LLaMA, Claude 3.5)
- OpenAI GPT-4o-mini support
- Google Gemini integration
- Excel export functionality
- Batch processing capabilities

