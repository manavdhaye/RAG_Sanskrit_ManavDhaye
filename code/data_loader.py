# import re
# from typing import List
# from langchain.schema import Document

# import re
# from typing import List
# from langchain_core.documents import Document

# def load_custom_sanskrit_docs(file_path: str) -> List[Document]:
#     print(f"Loading custom documents from {file_path}...")
#     documents = []
    
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#             print(f" > DEBUG: Successfully read {len(content)} characters from the file.")
#     except FileNotFoundError:
#         print(f"Error: File not found at {file_path}")
#         return []
            
#     # Added re.IGNORECASE just in case it says [Source: 1] instead of 
#     pattern = re.compile(r'\\s*(.*?)(?=\|$)', re.DOTALL | re.IGNORECASE)
#     matches = pattern.findall(content)
    
#     print(f" > DEBUG: Found {len(matches)} source tags matching the pattern.")
    
#     for source_id, text in matches:
#         clean_text = text.strip()
#         clean_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', clean_text)
        
#         if clean_text:
#             doc = Document(
#                 page_content=clean_text,
#                 metadata={"source_id": int(source_id)}
#             )
#             documents.append(doc)
            
#     print(f"Successfully loaded {len(documents)} chunks from the document.")
#     return documents

# if __name__ == "__main__":
#     # Quick test
#     docs = load_custom_sanskrit_docs("data/Rag-docs.txt")
#     if docs:
#         print(f"Sample Document: {docs[1].page_content}")
#         print(f"Metadata: {docs[1].metadata}")


import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_custom_sanskrit_docs(file_path: str) -> List[Document]:
    print(f"Loading custom documents from {file_path}...")
    documents = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f" > DEBUG: Successfully read {len(content)} characters from the file.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
            
    # Step 1: A much more forgiving string split. 
    # It looks for the word "source:" regardless of brackets or upper/lowercase.
    parts = re.split(r'(?i)\[?source:\s*', content)    
    for part in parts:
        if not part.strip():
            continue # Skip empty sections
            
        # Extract the number at the start, ignoring any trailing brackets or invisible spaces
        match = re.match(r'^(\d+)\]?\s*(.*)', part, re.DOTALL)
        if match:
            source_id = match.group(1)
            text = match.group(2).strip()
            
            # Clean out emails
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
            
            if text:
                doc = Document(
                    page_content=text,
                    metadata={"source_id": int(source_id)}
                )
                documents.append(doc)
                
    # Step 2: SAFETY FALLBACK
    # If the tags were completely lost in the Word-to-TXT conversion, 
    # chunk the document normally so the RAG pipeline still works!
    if len(documents) == 0:
        print(" > WARNING: Could not find any source tags. Using standard fallback chunking...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, 
            chunk_overlap=50,
            separators=["\n\n", "\n", "।", " "]
        )
        fallback_docs = text_splitter.create_documents([content])
        
        # Assign dummy source IDs so the LLM prompt still works
        for i, doc in enumerate(fallback_docs):
            doc.metadata = {"source_id": i + 1}
        documents = fallback_docs

    print(f"Successfully loaded {len(documents)} chunks from the document.")
    return documents

if __name__ == "__main__":
    # Quick test
    docs = load_custom_sanskrit_docs("../data/Rag-docs.txt")
    if docs:
        print(f"Sample Document: {docs[1].page_content}")
        print(f"Metadata: {docs[1].metadata}")
        print("documnet : ",docs)