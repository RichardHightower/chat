"""
Test script for the new search UI functionality.
Tests BM25, semantic, and hybrid search methods.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chat.rag.rag_service import RAGService
from vector_rag.model import ChunkResults


def test_search_methods():
    """Test all search methods with the RAG service."""
    
    # Initialize RAG service
    rag_service = RAGService()
    
    # Create or get a test project
    project = rag_service.get_or_create_project(
        name="Test Search UI",
        description="Testing BM25 and hybrid search"
    )
    
    print(f"Using project: {project.name} (ID: {project.id})")
    
    # Add some test documents if needed
    test_docs = [
        {
            "filename": "postgres_bm25.md",
            "content": """
            PostgreSQL BM25 Implementation Guide
            
            BM25 (Best Matching 25) is implemented in PostgreSQL using the full-text search
            capabilities. The tsvector column stores preprocessed searchable text, while
            ts_rank_cd() provides ranking similar to BM25.
            
            Key components:
            - tsvector: Stores tokenized and normalized text
            - tsquery: Represents search queries
            - GIN index: Accelerates full-text searches
            - ts_rank_cd(): Cover density ranking function
            """
        },
        {
            "filename": "vector_search.md", 
            "content": """
            Vector Search and Semantic Similarity
            
            Vector search uses embeddings to find semantically similar content. Unlike
            keyword search, it understands context and meaning. The process involves:
            
            1. Convert text to embeddings using models like sentence-transformers
            2. Store embeddings in a vector database
            3. Use cosine similarity to find related content
            4. Return results ranked by similarity score
            """
        },
        {
            "filename": "hybrid_search.md",
            "content": """
            Hybrid Search: Combining Vector and BM25
            
            Hybrid search combines the strengths of both semantic search and keyword
            matching. It's ideal for technical documentation where exact terms matter
            but context is also important.
            
            Benefits:
            - Captures exact technical terms with BM25
            - Understands semantic meaning with vectors
            - Configurable weights for different use cases
            - Better recall and precision than either method alone
            """
        }
    ]
    
    # Add documents if they don't exist
    existing_files = {f.name for f in rag_service.list_files(project.id)}
    
    for doc in test_docs:
        if doc["filename"] not in existing_files:
            print(f"Adding document: {doc['filename']}")
            rag_service.add_document(
                project_id=project.id,
                filename=doc["filename"],
                content=doc["content"],
                metadata={"type": "documentation"}
            )
    
    # Test different search methods
    test_queries = [
        ("PostgreSQL tsvector implementation", "Technical keyword search"),
        ("how to find similar documents", "Semantic understanding query"),
        ("BM25 ranking algorithm", "Mixed technical and conceptual")
    ]
    
    print("\n" + "="*60)
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}' ({description})")
        print("-" * 60)
        
        # Test semantic search
        print("\n1. Semantic Search Results:")
        semantic_results = rag_service.api.search_text(
            project_id=project.id,
            query_text=query,
            page_size=3,
            similarity_threshold=0.5
        )
        
        if semantic_results and semantic_results.results:
            for i, result in enumerate(semantic_results.results):
                print(f"   [{i+1}] Score: {result.score:.3f} - {result.chunk.file_name}")
                print(f"       Preview: {result.chunk.content[:100]}...")
        else:
            print("   No results found")
        
        # Test BM25 search
        print("\n2. BM25 (Keyword) Search Results:")
        bm25_results = rag_service.search_bm25(
            project_id=project.id,
            query=query,
            top_k=3,
            rank_threshold=0.0
        )
        
        if bm25_results and bm25_results.results:
            for i, result in enumerate(bm25_results.results):
                print(f"   [{i+1}] Rank: {result.score:.3f} - {result.chunk.file_name}")
                print(f"       Preview: {result.chunk.content[:100]}...")
        else:
            print("   No results found")
        
        # Test hybrid search
        print("\n3. Hybrid Search Results (50/50 weights):")
        hybrid_results = rag_service.search_hybrid(
            project_id=project.id,
            query=query,
            top_k=3,
            vector_weight=0.5,
            bm25_weight=0.5
        )
        
        if hybrid_results and hybrid_results.results:
            for i, result in enumerate(hybrid_results.results):
                print(f"   [{i+1}] Score: {result.score:.3f} - {result.chunk.file_name}")
                # Check for individual scores in metadata
                if "_scores" in result.chunk.metadata:
                    scores = result.chunk.metadata["_scores"]
                    print(f"       (Vector: {scores.get('vector', 0):.3f}, BM25: {scores.get('bm25', 0):.3f})")
                print(f"       Preview: {result.chunk.content[:100]}...")
        else:
            print("   No results found")
    
    print("\n" + "="*60)
    print("\nSearch method testing complete!")
    
    # Test the UI integration
    print("\n\nTo test the UI:")
    print("1. Run: poetry run streamlit run src/chat/app.py")
    print("2. Go to the 'RAG' tab and select the 'Test Search UI' project")
    print("3. Go to the 'Search' tab")
    print("4. Toggle 'Show search in main area'")
    print("5. Try different search methods and queries")
    print("\nFeatures to test:")
    print("- Semantic search with similarity threshold")
    print("- BM25 keyword search")
    print("- Hybrid search with adjustable weights")
    print("- Metadata tree view in expandable sections")
    print("- 'Send to Chat' functionality")
    print("- Tabular results view with sortable columns")


if __name__ == "__main__":
    test_search_methods()