# Gemini CLI Chatbot

A simple command-line chatbot built using Python and the Gemini API.

## Features

* Takes user input from the terminal
* Sends prompts to Gemini
* Receives AI-generated responses
* Displays responses in the terminal
* Runs continuously until the user types `bye`

---

## Technologies Used

* Python
* Gemini API
* google-generativeai
* python-dotenv

---

## Project Structure

```text
project/
│
├── chatbot.py
├── .env
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install google-generativeai
pip install python-dotenv
```

---

## Environment Variables

Create a `.env` file in the project root directory:

```env
API_KEY=YOUR_GEMINI_API_KEY
```

Replace `YOUR_GEMINI_API_KEY` with your actual Gemini API key.

---

## Running the Application

```bash
python chatbot.py
```

Example:

```text
You: What is Artificial Intelligence?
Bot: Artificial Intelligence is the simulation of human intelligence by machines.
```

Exit the chatbot:

```text
You: bye
```

---

## Workflow

```text
User Input
    ↓
Gemini API
    ↓
Model Response
    ↓
Terminal Output
```

---

## Learning Outcomes

This project demonstrates:

* Using environment variables securely
* Connecting a Python application to an LLM API
* Sending prompts to Gemini
* Receiving and displaying model responses
* Building a basic command-line chatbot

---

## Future Improvements

* Add conversation memory
* Read and process PDF documents
* Integrate LangChain
* Build a document question-answering system
* Add a graphical user interface (GUI)

```
```
