import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# Streamlit config
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜"
)
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Sidebar input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")


# Function to load YouTube transcript as Document
def load_youtube_transcript_as_docs(video_url):
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = "\n".join([entry.get("text", "") for entry in transcript])
        return [Document(page_content=full_text)]
    except Exception as e:
        raise RuntimeError(f"Transcript Error: {e}")


# Summarization logic
if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YT video or website")
    else:
        llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

        try:
            with st.spinner("Fetching and summarizing content..."):
                # Load content
                if "youtube.com" in generic_url:
                    docs = load_youtube_transcript_as_docs(generic_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"},
                    )
                    docs = loader.load()

                # Split into chunks to avoid token limit errors
                text_splitter = CharacterTextSplitter(
                    separator="\n", chunk_size=500, chunk_overlap=50
                )
                split_docs = text_splitter.split_documents(docs)

                # Adjusted Prompts for map/reduce summarization (300 words)
                map_prompt = PromptTemplate(
                    input_variables=["text"],
                    template="""
You are a helpful assistant. Summarize the following text in approximately 300 words, ensuring the summary captures all the main points concisely.
{text}
""",
                )
                combine_prompt = PromptTemplate(
                    input_variables=["text"],
                    template="""
You are a helpful assistant. Combine the following summaries into a final summary that is approximately 300 words, capturing all the key points of the entire document:
{text}
""",
                )

                # Run summarization
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                )
                output_summary = chain.run(split_docs)
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")
