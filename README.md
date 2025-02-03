# Research Copilot

This project is a Streamlit app that serves as an AI-powered assistant for summarizing and interacting with research papers. The assistant, built with LangChain and OpenAI's GPT-4, provides summaries, definitions for complex terms, examples, key points, and a conversational Q&A interface to help users better understand academic papers in minutes.

## Features
**Section Summarization:** Breaks down papers into sections like Abstract, Introduction, Methodology, Results, and Conclusion, summarizing each part for a quick overview.

**Terminology Definitions:** Extracts and explains complex terminology to make papers accessible.

**Key Notes:** Highlights the main points of the research paper.


**Conversational Q&A:** Engage in a chat-like conversation with the assistant for follow-up questions and deeper insights.

**Memory-Enabled Interaction:** Retains conversation context for natural, coherent responses across multiple questions.

## Technologies Used:

**LangChain:** For building the conversational chain with memory, using LLMChain and ConversationChain.

**Streamlit:** Simple web app framework to display summaries, definitions, and the conversational chatbot interface.

**OpenAI's GPT-4o:** The language model behind the chatbot’s responses and paper processing tasks.

## Getting Started

**Prerequisites**
* Python 3.8 or higher
* API key for OpenAI (for GPT-4 access)
Installation
Clone the repository:

## Installation

1. Clone the repository:

```
git clone https://github.com/sricharan245/Research-Copilot-App.git
cd Research-Copilot-App
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up OpenAI API key: Set your OpenAI API key as an environment variable.
```
export OPENAI_API_KEY="your-openai-api-key"
```

4. Run the app:

```
streamlit run app.py

```

## Usage
1. Upload a PDF: Click on "Upload PDF" and select a research paper.
2. Explore Summaries and Explanations:
* Summaries of key sections
* Terminology definitions
* Key notes

3. Conversational Q&A: Enter questions in the Q&A chat section to interact with the assistant, retaining context for follow-up questions.

## Screenshots
<img width="832" alt="image" src="https://github.com/user-attachments/assets/dc5bd3be-f89c-421f-ae67-4025c703bea3">
<img width="832" alt="image (1)" src="https://github.com/user-attachments/assets/4a80bc18-58b3-438a-abdd-3aa9634cb075">
<img width="832" alt="image (2)" src="https://github.com/user-attachments/assets/3429894f-85a3-4f59-b561-f9009806485c">
<img width="832" alt="image (3)" src="https://github.com/user-attachments/assets/30fbefc2-027d-40c6-a3c5-6758024e3e7d">


## License
This project is licensed under the MIT License.

## Acknowledgments
* LangChain for conversational chains with memory
* OpenAI for the GPT-4 language model
* Streamlit for the web app framework
