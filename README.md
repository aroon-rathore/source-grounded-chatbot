🤖 Source-Grounded Chatbot
A chatbot that answers questions using information extracted from web pages you provide. It processes URLs line-by-line and gives answers with source citations.

🚀 Quick Start
1. Install Requirements

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
2. Get Free API Key
Go to console.groq.com, sign up for free, and copy your API key.

3. Run the Chatbot

streamlit run simple_chatbot.py
📦 Requirements
See requirements.txt for the complete list. Main packages:
streamlit - Web interface
groq - Free AI API
beautifulsoup4 - Web scraping
scikit-learn - Text search

💡 How to Use
Open the app in your browser (usually http://localhost:8501)
Enter your Groq API key in the sidebar
Paste URLs (one per line) in the main area
Click "Process URLs" to extract content
Ask questions about the extracted information

Example URLs:
text
https://en.wikipedia.org/wiki/Elon_Musk
https://en.wikipedia.org/wiki/Artificial_intelligence
https://www.linkedin.com/in/williamhgates/
🔧 How It Works
Extracts text from web pages, line by line
Chunks the text for better processing
Searches for relevant information when you ask a question
Answers using only the extracted content
Shows sources (URL and line numbers)

🐛 Troubleshooting
"Module not found": Run pip install -r requirements.txt
API key error: Get a new key from console.groq.com
Webpage not loading: Try a different, publicly accessible URL
Port already in use: Run streamlit run simple_chatbot.py --server.port 8502

📁 Project Structure

source-grounded-chatbot/
├── simple_chatbot.py      # Main chatbot code
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore           # Files to ignore in Git
🆘 Need Help?
If the chatbot isn't working:
Make sure all packages are installed
Check your internet connection
Try a different URL (Wikipedia pages work well)
Get a fresh API key from Groq

Note: This chatbot uses information from the web pages you provide. Always verify important information from original sources.