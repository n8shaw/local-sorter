import os
import shutil
from typing import TypedDict, List
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
INPUT_DIR = "./input_dump"
OUTPUT_DIR = "./sorted_files"
DB_DIR = "./chroma_db"

# User Context
CONTEXT = """
The user is Nathan (Nate) Shaw, a 23-year-old student at BYU (Provo, UT) studying Information Systems (Masters/Bachelors).
He works as a Student Software Developer at the BYU Library and will be a Data Engineer at USAA in Phoenix (starting 2026).
Interests: AI/ML (Python, RAG), Web Dev (React, Django), Cooking (Sourdough, Mexican), Gaming (Minecraft, Clash Royale), One Piece, and Investing.
"""

RULES = """
1. If the 'History' shows a clear pattern (e.g. similar files went to "School/CS555"), use that.
2. If no history, invent a logical folder structure based on User Context (e.g. "Work/USAA", "Personal/Recipes", "School/General").
3. Return ONLY the folder path (e.g. "School/IS_566"). Do not include the output directory prefix.
"""

# --- 1. MEMORY SETUP (ChromaDB) ---

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    collection_name="sorterra_memory",
    embedding_function=embeddings,
    persist_directory=DB_DIR
)

# --- 2. STATE DEFINITION ---
class SortingState(TypedDict):
    file_path: str
    filename: str
    summary: str
    similar_files_context: str  # Text description of what we found in DB
    target_folder: str

# --- 3. TOOLS / FUNCTIONS ---

def read_file_content(file_path):
    """Safe read function."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def get_similar_files(summary: str):
    """Queries Chroma for top 3 similar files."""
    results = vector_store.similarity_search(summary, k=3)
    if not results:
        return "No similar files found in history."
    
    context_str = ""
    for doc in results:
        folder = doc.metadata.get("folder", "Unknown")
        name = doc.metadata.get("filename", "Unknown")
        context_str += f"- Found similar file '{name}' stored in folder: '{folder}'\n"
    return context_str

# --- 4. NODES ---

def summarizer_node(state: SortingState):
    """Node 1: Read and Summarize (gemma2:2b)"""
    print(f"\n--- Processing: {state['filename']} ---")
    content = read_file_content(state["file_path"])
    
    llm_light = ChatOllama(model="gemma2:2b", temperature=0)
    
    prompt = f"""
    You are a file analyzer. Summarize the following file content in 2-3 sentences. 
    Focus on keywords relevant to the user: {CONTEXT}.
    
    File Content:
    {content[:3000]} # Truncate for token limits if necessary
    """
    
    res = llm_light.invoke(prompt)
    print(f"Summary generated.")
    return {"summary": res.content}

def memory_node(state: SortingState):
    """Node 2: Query Vector DB"""
    summary = state["summary"]
    context = get_similar_files(summary)
    print(f"Memory Context: {context.strip()}")
    return {"similar_files_context": context}

def thinker_node(state: SortingState):
    """Node 3: Decide Destination (phi3.5)"""
    llm_thinker = ChatOllama(model="phi3.5", temperature=0)
    
    prompt = f"""
    You are an intelligent file sorter.
    
    User Context: {CONTEXT}
    
    Task: Decide the best folder category for the current file.
    
    Current File Summary:
    {state['summary']}
    
    History of Similar Files:
    {state['similar_files_context']}
    
    Instructions:
    {RULES}
    """
    
    res = llm_thinker.invoke(prompt)
    target_folder = res.content.strip().replace('"', '').replace("'", "")
    print(f"Decision: {target_folder}")
    return {"target_folder": target_folder}

def executor_node(state: SortingState):
    """Node 4: Move & Memorize"""
    filename = state["filename"]
    target_folder_name = state["target_folder"]
    full_target_dir = os.path.join(OUTPUT_DIR, target_folder_name)
    
    # 1. Create Dir
    os.makedirs(full_target_dir, exist_ok=True)
    
    # 2. Move File
    dst_path = os.path.join(full_target_dir, filename)
    shutil.move(state["file_path"], dst_path)
    
    # 3. Update Memory (The Learning Step)
    vector_store.add_texts(
        texts=[state["summary"]],
        metadatas=[{"folder": target_folder_name, "filename": filename}]
    )
    
    print(f"MOVED {filename} -> {target_folder_name}")
    return {} # End of state

# --- 5. GRAPH CONSTRUCTION ---
workflow = StateGraph(SortingState)

workflow.add_node("summarizer", summarizer_node)
workflow.add_node("memory", memory_node)
workflow.add_node("thinker", thinker_node)
workflow.add_node("executor", executor_node)

workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", "memory")
workflow.add_edge("memory", "thinker")
workflow.add_edge("thinker", "executor")
workflow.add_edge("executor", END)

app = workflow.compile()

# --- 6. MAIN EXECUTION LOOP ---
def run_sorting():
    # Ensure dirs exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    
    if not files:
        print("No files found in input directory.")
        return

    print(f"Found {len(files)} files. Starting Sorting...")
    
    for file in files:
        file_path = os.path.join(INPUT_DIR, file)
        initial_state = {
            "file_path": file_path, 
            "filename": file,
            "summary": "", 
            "similar_files_context": "", 
            "target_folder": ""
        }
        
        # Run the graph for this file
        app.invoke(initial_state)

if __name__ == "__main__":
    run_sorting()