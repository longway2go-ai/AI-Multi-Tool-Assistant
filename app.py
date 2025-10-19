import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Annotated
import os
import operator
import requests
import tempfile
from typing_extensions import TypedDict
from duckduckgo_search import DDGS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Multi-Tool Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ENHANCED CSS STYLING
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    .stTabs [role="tab"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 8px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 500;
        transition: all 0.3s ease;
        margin-right: 5px;
    }
    
    .stTabs [role="tab"]:hover {
        background: rgba(51, 65, 85, 0.9);
        border-color: #3b82f6;
        color: #60a5fa;
    }
    
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: #ffffff;
        border-color: #60a5fa;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .tool-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        margin: 0.4rem 0.4rem 0.4rem 0;
        border-radius: 20px;
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(96, 165, 250, 0.5);
    }
    
    .tool-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    .doc-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        margin: 0.4rem 0.4rem 0.4rem 0;
        border-radius: 20px;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        border: 1px solid rgba(16, 185, 129, 0.5);
    }
    
    .api-section {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(100, 116, 139, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stChatMessage {
        background: transparent;
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stChatMessage.user {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        margin-left: auto;
        margin-right: 0;
        max-width: 80%;
    }
    
    .stChatMessage.assistant {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
        border-left: 3px solid #10b981;
        border-radius: 8px;
        margin-right: auto;
        margin-left: 0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    .stTextInput>div>div>input {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(100, 116, 139, 0.3);
        color: #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
    }
    
    .stChatInputContainer {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebarContent"] {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        border-right: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    .sidebar-header {
        color: #60a5fa;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .success-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        color: #86efac;
    }
    
    .warning-message {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(234, 179, 8, 0.1) 100%);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        color: #fca5a5;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(100, 116, 139, 0.3), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# SESSION STATE INIT
def init_session_state():
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = []
    if 'threads' not in st.session_state:
        thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.threads = {
            thread_id: {
                "name": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "created_at": datetime.now()
            }
        }
        st.session_state.current_thread = thread_id
    if 'current_thread' not in st.session_state:
        st.session_state.current_thread = list(st.session_state.threads.keys())[0]


# TOOLS
@tool
def web_search(query: str) -> str:
    """Search the web for general information and trending topics."""
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=5)
        if not results:
            return "No search results found."
        out = f"🔍 Search Results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', 'No link')
            out += f"{i}. **{title}**\n{body}\n🔗 {href}\n\n"
        return out
    except Exception as e:
        return f"Search error: {str(e)}"


@tool
def get_latest_news(topic: str = "world", region: str = "us") -> str:
    """Fetch latest real-time news via NewsData.io free API."""
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        return "⚠ Please set NEWSDATA_API_KEY (get it free from https://newsdata.io)."
    try:
        url = f"https://newsdata.io/api/1/news?apikey={api_key}&q={topic}&country={region.lower()}&language=en"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "results" not in data or not data["results"]:
            return "No current news found."
        out = f"📰 Latest {topic.title()} News ({region.upper()}):\n\n"
        for i, article in enumerate(data["results"][:5], 1):
            out += f"{i}. {article['title']}\n{article.get('description','')}\nSource: {article.get('source_id','Unknown')} | {article.get('pubDate','Recent')}\n\n"
        return out
    except Exception as e:
        return f"News API error: {str(e)}"


@tool
def get_sports_updates(sport: str = "football", league: str = "") -> str:
    """Get live sports updates via API-Sports free plan."""
    api_key = os.getenv("SPORTS_API_KEY")
    if not api_key:
        return "⚠ Please set SPORTS_API_KEY (get it free from https://api-sports.io)."
    try:
        url = "https://v3.football.api-sports.io/fixtures?live=all" if sport.lower() == "football" else f"https://v1.all-sports-api.com/?method=get_events&s={sport.lower()}&APIkey={api_key}"
        headers = {"x-apisports-key": api_key}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        if sport.lower() == "football" and data.get("response"):
            results = data["response"]
            out = "⚽ Live Matches:\n\n"
            for m in results[:5]:
                home = m["teams"]["home"]["name"]
                away = m["teams"]["away"]["name"]
                hgoals = m["goals"]["home"]
                agoals = m["goals"]["away"]
                time = m["fixture"]["status"]["elapsed"]
                out += f"{home} {hgoals} - {agoals} {away} ({time} mins)\n\n"
            return out
        return "No live matches available."
    except Exception as e:
        return f"Sports API error: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Real-time weather via free Open-Meteo API."""
    try:
        from geopy.geocoders import Nominatim
        geo = Nominatim(user_agent="weather_agent", timeout=10)
        loc = geo.geocode(city, timeout=10)
        if not loc:
            return f"City '{city}' not found. Please check the spelling."
        lat, lon = loc.latitude, loc.longitude
        
        r = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true",
            timeout=15
        )
        r.raise_for_status()
        data = r.json().get("current_weather", {})
        
        if not data:
            return "Weather data unavailable."
        
        temp = data.get('temperature', 'N/A')
        wind = data.get('windspeed', 'N/A')
        time = data.get('time', 'N/A')
        
        return f"🌤 **{city.title()}**: {temp}°C | Wind: {wind} km/h | Time: {time}"
    except requests.exceptions.Timeout:
        return "⏱️ Weather API timeout. Please try again in a moment."
    except requests.exceptions.ConnectionError:
        return "🔌 Connection error. Please check your internet connection."
    except Exception as e:
        return f"⚠️ Weather error: {str(e)}"


@tool
def get_crypto_price(crypto: str = "bitcoin") -> str:
    """Fetch live cryptocurrency data via CoinGecko."""
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={crypto.lower()}&vs_currencies=usd,eur&include_24hr_change=true", timeout=10)
        data = r.json().get(crypto.lower())
        if not data:
            return "No crypto data."
        trend = "📈" if data.get("usd_24h_change", 0) > 0 else "📉"
        return f"₿ {crypto.title()}: ${data.get('usd')} / €{data.get('eur')}\n{trend} 24h Change: {data.get('usd_24h_change',0):.2f}%"
    except Exception as e:
        return f"Crypto data error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """Fetch real-time stock data via Alpha Vantage free API."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return "⚠ Please set ALPHAVANTAGE_API_KEY (get it free from https://www.alphavantage.co)."
    try:
        r = requests.get(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}", timeout=10)
        data = r.json()
        quote = data.get("Global Quote", {})
        if not quote:
            return "Stock data unavailable."
        return (f"📈 {symbol.upper()} Price: ${quote.get('05. price','N/A')}\n"
                f"Change: {quote.get('09. change','N/A')} ({quote.get('10. change percent','N/A')})\n"
                f"High: {quote.get('03. high','N/A')} | Low: {quote.get('04. low','N/A')}")
    except Exception as e:
        return f"Stock error: {str(e)}"


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Free exchange rate conversion."""
    try:
        data = requests.get(f"https://open.er-api.com/v6/latest/{from_currency.upper()}", timeout=10).json()
        rate = data["rates"].get(to_currency.upper())
        if not rate:
            return "Invalid currency code."
        converted = amount * rate
        return f"💱 {amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()} (Rate {rate:.4f})"
    except Exception as e:
        return f"Conversion error: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """Safe math calculator."""
    try:
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Invalid chars."
        return f"🧮 {expression} = {eval(expression, {'__builtins__': {}}, {})}"
    except Exception as e:
        return f"Calc error: {str(e)}"


@tool
def get_time_info(timezone: str = "UTC") -> str:
    """Get time info for timezone."""
    from datetime import datetime
    import pytz
    try:
        now = datetime.now(pytz.timezone(timezone))
        return f"🕒 {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Time error: {str(e)}"


@tool
def search_documents(query: str) -> str:
    """Search uploaded RAG documents."""
    if st.session_state.vectorstore is None:
        return "Upload documents first."
    docs = st.session_state.vectorstore.similarity_search(query, k=3)
    if not docs:
        return "No relevant info found."
    out = "📚 Found info:\n\n"
    for i, doc in enumerate(docs, 1):
        out += f"{i}. {doc.page_content}\n"
    return out


# AGENT SETUP
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


def create_agent(api_key: str):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key)
    tools = [
        get_latest_news, get_sports_updates, get_weather,
        get_crypto_price, get_stock_price, convert_currency,
        calculate, get_time_info, search_documents, web_search
    ]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        resp = llm_with_tools.invoke(state["messages"])
        return {"messages": [resp]}

    tool_node = ToolNode(tools)
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")

    workflow.add_conditional_edges("agent", lambda s: "tools" if getattr(s["messages"][-1], 'tool_calls', None) else "end",
                                   {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=MemorySaver())


# DOC UPLOAD
def process_uploaded_files(files, api_key):
    documents = []
    for f in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{f.name.split('.')[-1]}") as t:
                t.write(f.getvalue())
                path = t.name
            if f.name.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif f.name.endswith('.txt'):
                loader = TextLoader(path)
            elif f.name.endswith('.docx'):
                loader = Docx2txtLoader(path)
            else:
                continue
            docs = loader.load()
            for d in docs:
                d.metadata['source'] = f.name
            documents.extend(docs)
            os.unlink(path)
        except Exception as e:
            st.warning(f"Error loading {f.name}: {str(e)}")
            continue
    
    if not documents:
        st.error("No documents loaded. Please check file formats.")
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    
    if not splits:
        st.error("No content extracted from documents.")
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return FAISS.from_documents(splits, embeddings)


# MAIN APP
def main():
    init_session_state()
    
    st.markdown('<h1 class="main-header">🤖 AI Multi-Tool Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time AI with Internet-Connected Tools</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<p class="sidebar-header">⚙️ Configuration</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="api-section">', unsafe_allow_html=True)
        st.markdown('<p style="color: #60a5fa; font-weight: 600;">API Keys</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            api_key = st.text_input("🔑 Google API Key", type="password", help="Gemini for LLM")
        with col2:
            st.markdown("[📍 Get Key](https://aistudio.google.com/app/apikey)", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            news_key = st.text_input("📰 NewsData API Key", type="password", help="Real-time news API")
        with col2:
            st.markdown("[📍 Get Key](https://newsdata.io)", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            sports_key = st.text_input("⚽ API-Sports Key", type="password", help="Live sports updates")
        with col2:
            st.markdown("[📍 Get Key](https://api-sports.io)", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            stocks_key = st.text_input("📊 Alpha Vantage Key", type="password", help="Stock market data")
        with col2:
            st.markdown("[📍 Get Key](https://www.alphavantage.co)", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        if news_key:
            os.environ["NEWSDATA_API_KEY"] = news_key
        if sports_key:
            os.environ["SPORTS_API_KEY"] = sports_key
        if stocks_key:
            os.environ["ALPHAVANTAGE_API_KEY"] = stocks_key

        if not api_key:
            st.markdown('<div class="warning-message">⚠️ Please set your Google API key to continue.</div>', unsafe_allow_html=True)
            return

        st.markdown("---")
        st.markdown('<p class="sidebar-header">📚 Documents</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload (PDF/TXT/DOCX)", accept_multiple_files=True)
        if uploaded:
            if st.button("🔄 Process Documents", use_container_width=True):
                with st.spinner("🔍 Processing documents..."):
                    vectorstore = process_uploaded_files(uploaded, api_key)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.markdown('<div class="success-message">✅ Documents processed successfully!</div>', unsafe_allow_html=True)

    if 'agent' not in st.session_state or st.session_state.get('key') != api_key:
        st.session_state.agent = create_agent(api_key)
        st.session_state.key = api_key

    config = {"configurable": {"thread_id": st.session_state.current_thread}}
    state = st.session_state.agent.get_state(config)
    msgs = state.values.get("messages", []) if state.values else []

    for m in msgs:
        if isinstance(m, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.write(m.content)
        elif isinstance(m, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(m.content)

    if prompt := st.chat_input("💬 Ask me anything..."):
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            resp = ""
            # 🔧 Enhanced system prompt with explicit tool-use instructions
            sys_msg = SystemMessage(content="""
                You are an advanced conversational AI assistant, inspired by GPT-style chat models.
                You combine the reasoning power of Gemini with the personality, fluency, and helpfulness of ChatGPT.

                Your goals:
                1. Give **rich, natural, and engaging** responses — like a human expert who explains clearly.
                2. Always provide **helpful context**, examples, and insights beyond the bare answer.
                3. Offer **follow-up suggestions** to continue the conversation (3–4 ideas).
                4. If a user seems uncertain or vague, help them refine their request.
                5. Keep a friendly, professional tone — not robotic or repetitive.
                6. When you use a tool (like web_search or get_weather), **summarize** results conversationally, not as raw data.
                7. Always speak in complete paragraphs unless listing key points.

                Example style:
                User: "What's happening online today?"
                Assistant: "Here are some trending stories from across the web — from tech breakthroughs to entertainment buzz. I can dive deeper into any of these or check live updates for a specific topic. Would you like me to focus on world news, sports, or tech trends next?"

                At the end of each answer, include a short section like:
                💬 *You could ask next:* “Show me more details about ___” | “What do people think about this?” | “Find similar topics.”

                When vague or general prompts appear, use `web_search("latest trending topics")` or `get_latest_news("world")` automatically.
                """)


            input_msgs = [sys_msg, HumanMessage(content=prompt)]

            resp = ""
            message_placeholder = st.empty()

            # 🧠 Stream model response (with tools)
            for event in st.session_state.agent.stream(
                {"messages": input_msgs}, config=config, stream_mode="values"
            ):
                if "messages" in event and event["messages"]:
                    last = event["messages"][-1]
                    if isinstance(last, AIMessage) and last.content:
                        resp = last.content
                        message_placeholder.write(resp)

            # 🛠️ Fallback: If Gemini didn’t call a tool and response is vague, trigger web_search automatically
            if not resp or "cannot" in resp.lower() or "unable" in resp.lower():
                fallback_query = "latest trending topics on the internet"
                with st.spinner("🌐 Fetching live internet trends..."):
                    result = web_search(fallback_query)
                    message_placeholder.write(result if result else "⚠️ No results found.")



if __name__ == "__main__":
    main()