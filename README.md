# EverAI Simulator Backend

A FastAPI-based backend service for the EverAI Simulator platform that handles training simulations, modules, and training plans. The system provides APIs for managing customer service simulations, audio-to-script conversions, and training analytics.

## Features

- Training plan management
- Simulation creation and execution
- Audio-to-script conversion
- Real-time playback analytics
- Voice integration with Retell AI
- MongoDB integration for data persistence
- Azure OpenAI integration for script generation

## Prerequisites

- Python 3.11 or higher
- MongoDB instance
- Azure OpenAI account
- Deepgram API key
- Retell AI API key

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# MongoDB Configuration
mongo-url=your_mongodb_connection_string
db-name=everai_simulator

# API Keys
DEEPGRAM_API_KEY=your_deepgram_api_key
RETELL_API_KEY=your_retell_api_key

# Azure OpenAI Configuration
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_BASE_URL=your_azure_openai_base_url
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the FastAPI server:
```bash
python main.py
```

The server will start at `http://localhost:8000`

## Project Structure

```
├── api/
│   ├── controllers/    # API route handlers
│   └── schemas/       # Request/response models
├── domain/
│   ├── interfaces/    # Abstract base classes
│   ├── models/        # Domain models
│   ├── plugins/       # External service integrations
│   └── services/      # Business logic
├── infrastructure/
│   ├── database.py    # Database connection
│   └── repositories/  # Data access layer
└── main.py           # Application entry point
```