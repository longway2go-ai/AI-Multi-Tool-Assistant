# 🤖 AI Multi-Tool Assistant

A powerful, feature-rich Streamlit application that combines Google's Gemini AI with real-time internet-connected tools for news, weather, sports, stocks, crypto, currency conversion, and document search.

## ✨ Features

- **Real-Time Web Search** 🔍 - Search the internet using DuckDuckGo
- **Latest News** 📰 - Fetch current news by topic and region
- **Live Sports Updates** ⚽ - Get live match scores and sports events
- **Weather Information** 🌤️ - Real-time weather for any city
- **Cryptocurrency Prices** ₿ - Live crypto market data
- **Stock Market Data** 📈 - Real-time stock prices and analytics
- **Currency Conversion** 💱 - Free exchange rate conversions
- **Calculator** 🧮 - Safe mathematical expressions
- **Time Zone Info** 🕒 - Get time for any timezone
- **Document Search (RAG)** 📚 - Upload and search PDFs, TXT, DOCX files
- **Beautiful Dark UI** 🎨 - Modern gradient design with smooth animations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone or download the project**
```bash
cd your_project_directory
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dependencies

```
streamlit==1.28.0
langchain==0.1.0
langchain-google-genai==0.0.5
langgraph==0.0.1
duckduckgo-search==3.9.2
geopy==2.3.0
requests==2.31.0
faiss-cpu==1.7.4
PyPDF2==3.0.1
python-docx==0.8.11
```

Install all at once:
```bash
pip install streamlit langchain langchain-google-genai langgraph duckduckgo-search geopy requests faiss-cpu PyPDF2 python-docx
```

## 🔑 API Keys Configuration

The app requires various API keys for full functionality. Get them free from:

| Service | Purpose | Get Key |
|---------|---------|---------|
| **Google Gemini** | AI Model & Embeddings | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| **NewsData.io** | Latest News | [newsdata.io](https://newsdata.io) |
| **API-Sports** | Live Sports Updates | [api-sports.io](https://api-sports.io) |
| **Alpha Vantage** | Stock Market Data | [alphavantage.co](https://www.alphavantage.co) |

### Setup Instructions

1. Run the app: `streamlit run app.py`
2. Look at the **Configuration** section in the left sidebar
3. Click the **📍 Get Key** links next to each API field
4. Paste your API keys in the input fields
5. Keys are stored in session (not saved permanently)

## 🛠️ How to Use

### 1. **Ask Questions**
```
"What's the weather in Kolkata?"
"Show me trending topics today"
"What's the stock price of AAPL?"
```

### 2. **Upload Documents**
- Click **📄 Upload Docs** in the sidebar
- Select PDF, TXT, or DOCX files
- Click **🔄 Process Documents**
- Ask questions about your documents

### 3. **Available Commands**
```
Weather:        "What's the weather in Paris?"
News:           "Show me latest tech news from US"
Sports:         "Who's playing football today?"
Stocks:         "Tesla stock price"
Crypto:         "Bitcoin price in EUR"
Convert:        "Convert 100 USD to INR"
Calculate:      "What's 25 * 4 + 10?"
Time:           "What time is it in Tokyo?"
Search:         "What's trending on the internet?"
Documents:      "Summarize the document I uploaded"
```

## 📁 Project Structure

```
project/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 UI Features

- **Dark Theme** - Easy on the eyes with gradient backgrounds
- **Glassmorphism** - Modern frosted glass effect on cards
- **Smooth Animations** - Hover effects and transitions
- **Responsive Design** - Works on desktop and mobile
- **Real-time Streaming** - See AI responses as they're generated
- **Color-Coded Tools** - Blue for AI tools, Green for documents

## 🔧 Customization

### Change Model
Edit line with `ChatGoogleGenerativeAI`:
```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
```

Available models:
- `gemini-2.0-flash` - Fastest
- `gemini-1.5-pro` - Most capable
- `gemini-1.5-flash` - Balanced

### Add More Tools
Add new functions with `@tool` decorator:
```python
@tool
def my_tool(param: str) -> str:
    """Tool description"""
    # Your code here
    return result

# Add to tools list in create_agent()
```

### Modify UI Colors
Edit the CSS in the `st.markdown()` section with custom colors.

## ⚠️ Troubleshooting

### Weather Tool Not Working
```bash
pip install geopy
# Test: python -c "from geopy.geocoders import Nominatim; print('OK')"
```

### Document Upload Issues
- Ensure files are not corrupted
- Check file format (PDF, TXT, DOCX only)
- Large files may take time to process

### API Errors
- Verify API keys are correct
- Check rate limits (free tiers have limits)
- Ensure internet connection is stable

### DuckDuckGo Search Not Working
```bash
pip install --upgrade duckduckgo-search
```

## 📊 Performance Tips

1. **Use smaller document chunks** - Reduces processing time
2. **Cache API responses** - Avoid redundant calls
3. **Disable unused tools** - Remove from tools list if not needed
4. **Use Gemini Flash** - Faster responses for general queries

## 🔐 Security Notes

- **API keys are session-only** - Not stored to disk
- **No data persistence** - Chats are not saved between sessions
- **Safe calculator** - Uses sandboxed evaluation
- **Input validation** - All external inputs are validated

## 📝 Example Conversations

```
User: "What's trending today?"
AI: Uses web_search tool → Returns trending topics from DuckDuckGo

User: "Convert 500 USD to INR"
AI: Uses convert_currency tool → Returns live exchange rate

User: "What's Bitcoin price?"
AI: Uses get_crypto_price tool → Returns live crypto data

User: "Summarize my document"
AI: Uses search_documents tool → Searches uploaded RAG documents
```

## 🤝 Contributing

Feel free to:
- Add new tools
- Improve UI styling
- Optimize performance
- Report bugs

## 📄 License

This project is open source and available for personal and commercial use.

## 🙋 Support

### Common Issues

**Q: App runs but no response from model?**
- Check Google API key is valid
- Ensure internet connection is active

**Q: Tools not being used?**
- Verify API keys are set in sidebar
- Check that tool description is clear

**Q: Slow responses?**
- Use gemini-2.0-flash model
- Reduce document chunk size
- Close other applications

**Q: Can't upload documents?**
- Check file size (convert large files to text)
- Ensure format is PDF/TXT/DOCX
- Try individual files first

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Chat history saving
- [ ] Voice input/output
- [ ] Image analysis
- [ ] Real-time notifications
- [ ] Database integration
- [ ] User authentication
- [ ] Advanced analytics

## 📞 Contact & Support

For issues, questions, or feature requests, please reach out or check the documentation.

---

**Made with ❤️ using Streamlit, LangChain, and Google Gemini**

Last Updated: October 2025