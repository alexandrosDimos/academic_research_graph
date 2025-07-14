from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from typing import TypedDict, List, Optional
from IPython.display import Image
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()

filename = "Swarms of Unmanned Aerial Vehicles.pdf"
loader = PyPDFLoader(file_path=filename)
docs = []
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=30,
    add_start_index=True)

all_splits = text_splitter.split_documents(docs)

print(f"Split the pdf into {len(all_splits)} sub-documents.")



#embeddings = OllamaEmbeddings(model="llama3.2",)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embedding=embeddings)


document_ids = vector_store.add_documents(documents=all_splits)


custom_prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks and your expertise is academic research. Use the following pieces of retrieved context to answer the question. If you don't know the answer, 
    just say that you don't know. If any feedback is provided take it under consideration and reply based on how the feedback applies to the question. You answer mainly to the question when context and feedback are None.
    Question: {question} 
    Context: {context}
    Feedback: {feedback} 
    Answer:
"""
)


#prompt = hub.pull('rlm/rag-prompt')
llm = ChatOllama(model="llama3.2", temperature=0.5)
#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    feedback: Optional[str]


def retrieve(state: State):
    print("--Retrieving Information--")
    question = state['question']
    retrieved_docs = vector_store.similarity_search_with_score(query=question,k=4)
    #for i, (doc, score) in enumerate(retrieved_docs):
    #    print(f"\nDocument {i+1} (score: {score:.4f}):")
    #    print(doc.page_content[:500])
    #state['feedback'] = None
    docs_only = [doc for doc, score in retrieved_docs]
    return {'context': docs_only, 'feedback': None}

def generate(state: State):
    print("--Generating Response--")
    #docs_contents =  "\n\n".join([doc.page_content for doc in state["context"]])
    docs_contents =  "\n\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(state["context"])])
    #print("----------------DOC CONTENTS----------------")
    #print(docs_contents)
    messages = custom_prompt.invoke({'question': state['question'], 'context': state['context'], 'feedback':state['feedback']})
    response = llm.invoke(messages)
    return {'answer': response}

def feedback_loop(state: State):
    print("--Feedback Loop--")
    # Display the current answer and prompt for feedback
    #print(f"Generated Answer: {state['answer']}")
    feedback = input("Do you approve this answer? (yes/no) or provide suggestions: ").strip()

    # Process feedback
    if feedback.lower() == "yes":
        return {'feedback': 'approved', 'answer': None}
    else:
        # Update the feedback state with user input
        
        return {'feedback': feedback, 'answer': None}


def feedback_check(state: State):
    if state['feedback'] != 'approved':
        state['feedback'] = None
        return "generate"
    
    if state['feedback'] == 'approved':
        return END

builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_node("feedback_loop", feedback_loop)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "feedback_loop")

builder.add_conditional_edges("feedback_loop", feedback_check)
#builder.add_edge("feedback_loop", lambda state: state['feedback'] == 'approved', END)

graph = builder.compile()

#print(graph.get_graph().draw_ascii())
#with open("graph.png", "wb") as f:
#    f.write(graph.get_graph().draw_mermaid_png())
#png  = Image(graph.get_graph().draw_mermaid_png())

#png.save()


def interactive_rag():
    print("Interactive RAG System. Type 'exit' to quit.")
    
    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            break

        # Create the initial state with the user's question
        state = {'question': question}
        
        # Stream responses from the graph
        for event in graph.stream(state, stream_mode='values'):
            answer = event.get('answer', '')
            if answer:
                print("\nAnswer: ")
                answer.pretty_print()

        print("\n--- End of response ---")

interactive_rag()