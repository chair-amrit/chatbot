# Gemini CLI Chatbot

A simple command-line chatbot built with Python and Google's Gemini API.

## Features

- Runs in the terminal
- Sends user messages to Gemini
- Keeps a conversation session while the app is running
- Skips empty messages
- Exits when the user types `bye`, `exit`, or `quit`
- Reads the API key from a `.env` file

## Project Structure

```text
chatbot/
|-- chatbot.py
|-- requirements.txt
|-- readme.md
|-- .env
`-- venv/
```

## Requirements

- Python 3.10 or newer
- Gemini API key

Python packages:

```text
google-generativeai>=0.8.0
python-dotenv>=1.0.0
```

## Setup

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

The `.env` file is ignored by Git so your real API key is not committed.

## Run

Using the active virtual environment:

```powershell
python chatbot.py
```

Or run directly with the virtual environment Python:

```powershell
venv\Scripts\python.exe chatbot.py
```

## Usage

Example:

```text
You: What is artificial intelligence?
Bot: Artificial intelligence is the ability of machines to perform tasks that usually require human intelligence.
```

Exit the chatbot with any of these:

```text
bye
exit
quit
```

## How It Works

```text
User input
-> chatbot.py
-> Gemini API
-> model response
-> terminal output
```

## Notes

This project currently uses the `google-generativeai` package. That package now shows a deprecation warning and Google recommends migrating to the newer `google.genai` package in a future upgrade.

## Possible Next Upgrades

- Add `/help` and `/clear` commands
- Add configurable model settings
- Save chat history to a file
- Add document or PDF question answering
- Migrate from `google-generativeai` to `google.genai`
