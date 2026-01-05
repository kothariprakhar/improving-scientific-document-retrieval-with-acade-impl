import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import random
import numpy as np

# ==========================================
# 1. Mock Components (LLM & Taxonomy)
# ==========================================

class MockLLM:
    """
    Simulates a Large Language Model (e.g., GPT-4, Llama) used for 
    query generation and context expansion.
    In a real scenario, this would wrap an API call (OpenAI, vLLM, etc.).
    """
    def generate_query(self, document_text: str, concept: str) -> str:
        # Simulate generating a query specific to a concept
        return f"What is the role of {concept} in {document_text[:20]}...?"

    def generate_answer_context(self, document_text: str, query: str) -> str:
        # Simulate extracting/generating a concise snippet/answer from the doc
        return f"Snippet response to '{query}': The concept appears significant..."

class ConceptExtractor:
    """
    Simulates the Academic Concept Index construction.
    In the paper, this involves extracting entities and mapping to a taxonomy.
    """
    def extract(self, text: str) -> List[str]:
        # Dummy logic: extract capitalized words as 'concepts' for demonstration
        words = text.split()
        concepts = [w.strip('.,') for w in words if w[0].isupper() and len(w) > 4]
        return list(set(concepts))[:3] # Return top 3 unique mock concepts

# ==========================================
# 2. CCQGen: Concept Coverage-based Query Gen
# ==========================================

class CCQGen:
    """
    Implements the logic for generating synthetic queries conditioned on
    specific academic concepts to ensure broad coverage.
    """
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def generate(self, document: str, concepts: List[str]) -> List[Dict]:
        """
        Generates (Query, Concept) pairs.
        Paper Logic: P(q | d, c)
        """
        generated_data = []
        for concept in concepts:
            # Condition generation on the specific concept
            syn_query = self.llm.generate_query(document, concept)
            generated_data.append({
                "query": syn_query,
                "target_concept": concept,
                "document": document
            })
        return generated_data

# ==========================================
# 3. CCExpand: Concept-focused Context Augmentation
# ==========================================

class CCExpand:
    """
    Implements the logic for augmenting the document representation
    with generated contexts (answers) to the CCQGen queries.
    """
    def __init__(self, llm: MockLLM):
        self.llm = llm

    def expand_document(self, document: str, generated_queries: List[str]) -> str:
        """
        Augments document with snippets.
        Doc_final = Doc + [SEP] + Snippet_1 + [SEP] + Snippet_2 ...
        """
        augmented_parts = [document]
        
        for query in generated_queries:
            # Generate specific context/answer for the concept-based query
            context_snippet = self.llm.generate_answer_context(document, query)
            augmented_parts.append(context_snippet)
        
        # Join with a separator (using a simple string separator for this demo)
        return " [SEP] ".join(augmented_parts)

# ==========================================
# 4. Dense Retrieval Model (Bi-Encoder)
# ==========================================

class DenseRetriever(nn.Module):
    """
    Standard Bi-Encoder architecture (like DPR or Contriever).
    Encodes Queries and (Expanded) Documents into the same vector space.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        outputs = self.encoder(**inputs)
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        # Normalize for cosine similarity
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def forward(self, queries: List[str], docs: List[str]):
        q_embs = self.encode(queries)
        d_embs = self.encode(docs)
        return q_embs, d_embs

# ==========================================
# 5. Main Training Pipeline Simulation
# ==========================================

def run_experiment():
    print("--- Starting CCQGen and CCExpand Experiment ---")
    
    # Setup
    llm = MockLLM()
    extractor = ConceptExtractor()
    ccqgen = CCQGen(llm)
    ccexpand = CCExpand(llm)
    
    # Dummy Scientific Documents
    raw_documents = [
        "Deep Learning utilizes Neural Networks to optimize Loss Functions via Backpropagation.",
        "Transformer models rely on Self-Attention mechanisms and Positional Encodings for NLP.",
        "Graph Convolutional Networks aggregate features from neighbor nodes in a Topology."
    ]

    # Step 1: Concept Extraction & CCQGen (Data Generation)
    print("\n[1] Extracting Concepts and Generating Synthetic Queries (CCQGen)...")
    training_pairs = []
    expanded_docs_map = {} # Maps original doc index to expanded text

    for idx, doc in enumerate(raw_documents):
        concepts = extractor.extract(doc)
        print(f"   Doc {idx} Concepts: {concepts}")
        
        # Generate Queries conditioned on concepts
        gen_data = ccqgen.generate(doc, concepts)
        
        # Store for training (Query, Doc_Index)
        # In the paper, these are positive pairs for training
        for item in gen_data:
            training_pairs.append((item['query'], idx))
        
        # Step 2: CCExpand (Document Expansion)
        # Create an expanded version of the document to be used in the index
        queries_for_expansion = [item['query'] for item in gen_data]
        expanded_text = ccexpand.expand_document(doc, queries_for_expansion)
        expanded_docs_map[idx] = expanded_text

    print(f"\n[2] Document Expansion Complete. Example expanded doc:\n    {list(expanded_docs_map.values())[0][:150]}...")

    # Step 3: Train/Fine-tune Retrieval Model
    print("\n[3] Simulating Training Loop with Contrastive Loss...")
    retriever = DenseRetriever()
    optimizer = optim.AdamW(retriever.parameters(), lr=2e-5)
    
    # Prepare Batch
    # We simply take all generated pairs for this demo batch
    batch_queries = [p[0] for p in training_pairs]
    # The positive document for each query is the EXPANDED version of its source doc
    batch_docs = [expanded_docs_map[p[1]] for p in training_pairs]
    
    retriever.train()
    
    # Forward Pass
    q_vecs, d_vecs = retriever(batch_queries, batch_docs)
    
    # Contrastive Loss (In-batch negatives)
    # Similarity matrix: (Batch_Size, Batch_Size)
    scores = torch.matmul(q_vecs, d_vecs.T) * 20.0 # Scale by temperature inverse
    
    # Labels are the diagonal (0, 1, 2...) assuming 1-to-1 mapping in this batch construction
    # (Note: In reality, multiple queries map to same doc, so we'd mask correct docs, but diagonal is standard InfoNCE baseline)
    labels = torch.arange(len(batch_queries), device=scores.device)
    
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(scores, labels)
    
    print(f"   Training Batch Loss: {loss.item():.4f}")
    
    # Step 4: Inference/Retrieval Test
    print("\n[4] Performing Inference...")
    retriever.eval()
    test_query = "How does backpropagation work?"
    
    # Index all expanded documents
    all_expanded_docs = list(expanded_docs_map.values())
    with torch.no_grad():
        q_emb = retriever.encode([test_query])
        d_embs = retriever.encode(all_expanded_docs)
        
        # Dot product similarity
        sims = torch.matmul(q_emb, d_embs.T).squeeze()
        best_doc_idx = torch.argmax(sims).item()
    
    print(f"   Query: '{test_query}'")
    print(f"   Retrieved Doc Index: {best_doc_idx}")
    print(f"   Retrieved Content Start: {all_expanded_docs[best_doc_idx][:50]}...")

if __name__ == "__main__":
    run_experiment()
