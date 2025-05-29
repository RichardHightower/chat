"""
Search interface for RAG documents with support for multiple search methods.
Provides a tabular view of search results with metadata exploration.
"""

import streamlit as st
import pandas as pd
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from chat.rag.rag_service import RAGService
from vector_rag.model import ChunkResults

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def parse_metadata_filters(filter_str: str) -> Dict:
    """Parse metadata filter string into a dictionary.
    
    Args:
        filter_str: String like "type=toc, status=active, pages>10"
        
    Returns:
        Dictionary of filters for the vector-rag API
    """
    if not filter_str or not filter_str.strip():
        return {}
    
    filters = {}
    
    # Split by comma and process each filter
    for filter_part in filter_str.split(','):
        filter_part = filter_part.strip()
        if not filter_part:
            continue
            
        # Parse operators (=, >, <, >=, <=)
        if '>=' in filter_part:
            key, value = filter_part.split('>=', 1)
            key, value = key.strip(), value.strip()
            # Convert to number if possible
            try:
                value = float(value)
            except ValueError:
                pass
            filters[key] = {'$gte': value}
        elif '<=' in filter_part:
            key, value = filter_part.split('<=', 1)
            key, value = key.strip(), value.strip()
            try:
                value = float(value)
            except ValueError:
                pass
            filters[key] = {'$lte': value}
        elif '>' in filter_part:
            key, value = filter_part.split('>', 1)
            key, value = key.strip(), value.strip()
            try:
                value = float(value)
            except ValueError:
                pass
            filters[key] = {'$gt': value}
        elif '<' in filter_part:
            key, value = filter_part.split('<', 1)
            key, value = key.strip(), value.strip()
            try:
                value = float(value)
            except ValueError:
                pass
            filters[key] = {'$lt': value}
        elif '=' in filter_part:
            key, value = filter_part.split('=', 1)
            key, value = key.strip(), value.strip()
            # Try to convert to number or boolean
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            else:
                try:
                    # Try int first, then float
                    if '.' not in value:
                        value = int(value)
                    else:
                        value = float(value)
                except ValueError:
                    # Keep as string
                    pass
            filters[key] = value
    
    return filters


def extract_nested_value(data: Dict, path: str) -> Any:
    """Extract a value from nested dictionary using dot notation or bracket notation.
    
    Args:
        data: The dictionary to extract from
        path: Path like 'field.subfield' or 'field["sub field"]'
        
    Returns:
        The extracted value or None if not found
    """
    import re
    
    # Handle empty data
    if not data:
        return None
    
    # Split path by dots, but not dots inside brackets
    parts = re.split(r'\.(?![^[]*])', path)
    
    current = data
    for part in parts:
        if not current:
            return None
            
        # Handle bracket notation
        if '[' in part and ']' in part:
            # Extract key and index/key from brackets
            base_key = part[:part.index('[')]
            bracket_content = part[part.index('[') + 1:part.rindex(']')]
            
            # Remove quotes if present
            bracket_content = bracket_content.strip('"').strip("'")
            
            # Navigate to base key if present
            if base_key:
                if isinstance(current, dict) and base_key in current:
                    current = current[base_key]
                else:
                    return None
            
            # Handle the bracket content
            if isinstance(current, dict) and bracket_content in current:
                current = current[bracket_content]
            elif isinstance(current, list):
                try:
                    idx = int(bracket_content)
                    if 0 <= idx < len(current):
                        current = current[idx]
                    else:
                        return None
                except ValueError:
                    return None
            else:
                return None
        else:
            # Simple key access
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
    
    return current


def render_search_interface(rag_service: RAGService, project_id: int):
    """Render the search interface with multiple search method support.
    
    Args:
        rag_service: The RAG service instance
        project_id: Current project ID
    """
    st.header("📚 Document Search")
    
    # Initialize session state for custom columns if not exists
    if 'custom_columns' not in st.session_state:
        st.session_state.custom_columns = []
        print("DEBUG: Initialized custom_columns in session state")
    
    # Initialize session state for search results
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = None
        st.session_state.last_search_method = None
        st.session_state.last_search_query = None
        st.session_state.last_metadata_filters = None
    
    print(f"DEBUG: Current custom_columns in session state: {st.session_state.custom_columns}")
    print(f"DEBUG: Session state keys: {list(st.session_state.keys())}")
    
    # Search method selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_method = st.selectbox(
            "Search Method",
            ["Semantic Search", "BM25 (Keyword)", "Hybrid Search", "Metadata Query"],
            help="""
            - **Semantic Search**: Finds conceptually similar content
            - **BM25**: Exact keyword matching (best for technical terms)
            - **Hybrid**: Combines semantic and keyword search
            - **Metadata Query**: Direct metadata-based filtering (no scoring)
            """
        )
    
    with col2:
        # Initialize default weights
        vector_weight = 0.5
        bm25_weight = 0.5
        
        if search_method == "Hybrid Search":
            st.markdown("**Weight Configuration**")
            vector_weight = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Weight for semantic similarity (0-1)"
            )
            bm25_weight = 1.0 - vector_weight
            st.caption(f"BM25 Weight: {bm25_weight:.1f}")
    
    # Search input - make optional for metadata queries
    if search_method == "Metadata Query":
        search_query = st.text_input(
            "Content Search (optional)",
            placeholder="Optional: search within content using simple text matching",
            help="Leave empty to search by metadata only, or add text to also filter content"
        )
    else:
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter your search terms...",
            help="Type keywords or natural language queries"
        )
    
    # Handle quick filter application
    quick_filter_value = ""
    if 'quick_filter' in st.session_state:
        quick_filter_value = st.session_state.quick_filter
        del st.session_state.quick_filter  # Remove after using
    
    # Metadata filters - required for metadata queries
    if search_method == "Metadata Query":
        metadata_filters_str = st.text_input(
            "Metadata Filters (required)",
            value=quick_filter_value,
            placeholder="type=section, section=4. Clinical Results",
            help="""
            **Required for Metadata Query.** Filter by metadata fields:
            • type=section (load all sections)
            • type=paragraph, section=Introduction (paragraphs from intro)
            • document=report.pdf, page_number>10 (specific document pages)
            • author.department=Biostatistics (nested metadata)
            """
        )
    else:
        metadata_filters_str = st.text_input(
            "Metadata Filters (optional)",
            value=quick_filter_value,
            placeholder="type=toc, status=active, pages>10",
            help="""
            Filter by metadata fields. Examples:
            • type=toc (exact match)
            • type=section (exact match)
            • pages>10 (numeric comparison)
            • size>=1000 (file size filter)
            • Multiple: type=toc, pages>5
            """
        )
    
    # Quick filter buttons
    if st.session_state.get('last_search_results'):
        st.markdown("**Quick Filters:**")
        filter_cols = st.columns(5)
        
        # Different quick filters based on search method
        if search_method == "Metadata Query":
            quick_filters = [
                ("type=section", "All Sections"),
                ("type=paragraph", "All Paragraphs"),
                ("type=heading", "All Headings"),
                ("type=table", "All Tables"),
                ("", "Clear Filters")
            ]
        else:
            quick_filters = [
                ("type=toc", "TOC Only"),
                ("type=section", "Sections Only"),
                ("type=paragraph", "Paragraphs"),
                ("pages>10", "Large Docs"),
                ("", "Clear Filters")
            ]
        
        for i, (filter_str, label) in enumerate(quick_filters):
            with filter_cols[i]:
                if st.button(label, key=f"quick_filter_{i}"):
                    st.session_state.quick_filter = filter_str
                    # We'll handle this below
                    st.rerun()
    
    # Show metadata query examples if that method is selected
    if search_method == "Metadata Query":
        with st.expander("📖 Metadata Query Examples", expanded=False):
            st.markdown("""
            **Load specific document sections:**
            - `section=4. Clinical Results` - All chunks from section 4
            - `type=paragraph, section=Introduction` - Paragraphs from intro
            - `document=report.pdf, type=section` - All sections from specific document
            
            **Navigation and structure:**
            - `type=heading` - All document headings (for navigation)
            - `type=table` - All tables in the document
            - `type=figure` - All figures and charts
            
            **Filter by document properties:**
            - `author.name=Dr. Smith` - Content by specific author
            - `page_number>=10, page_number<=20` - Specific page range
            - `language=en` - English content only
            
            **Combine content search:**
            - Add text in "Content Search" field to also filter by content within metadata results
            """)
    
    # Initialize default values
    similarity_threshold = 0.7
    rank_threshold = 0.0
    
    # Advanced options expander
    with st.expander("Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            page_size = st.number_input(
                "Results per page",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )
            
            # Add performance tip
            if page_size > 20:
                st.warning("⚠️ Large result sets may be slow. Consider using smaller page sizes or higher thresholds.")
        
        with col2:
            if search_method in ["Semantic Search", "Hybrid Search"]:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="Minimum similarity score for semantic search"
                )
            
            if search_method in ["BM25 (Keyword)", "Hybrid Search"]:
                rank_threshold = st.slider(
                    "BM25 Rank Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help="Minimum BM25 rank score"
                )
            
            if search_method == "Metadata Query":
                st.info("💡 Results ordered by database storage order")
                st.markdown("**Tip:** Use `page_number` in filters to get document order")
                st.warning("⚠️ Note: Metadata queries return all matching results (no pagination)")
        
        with col3:
            show_metadata = st.checkbox("Show Metadata", value=True)
            # Don't show scores for metadata queries since they don't have relevance scores
            if search_method != "Metadata Query":
                show_scores = st.checkbox("Show Scores", value=True)
            else:
                show_scores = False
                st.info("ℹ️ Metadata queries return results in database order (no relevance scores)")
    
    # Search button - different requirements based on search method
    if search_method == "Metadata Query":
        search_disabled = not metadata_filters_str.strip()
        if search_disabled:
            st.warning("⚠️ Metadata filters are required for Metadata Query mode")
    else:
        search_disabled = not search_query
    
    if st.button("🔍 Search", type="primary", disabled=search_disabled):
        
        with st.spinner("Searching..."):
            try:
                # Parse metadata filters
                metadata_filters = parse_metadata_filters(metadata_filters_str)
                
                results = perform_search(
                    rag_service,
                    project_id,
                    search_query,
                    search_method,
                    page_size,
                    similarity_threshold,
                    rank_threshold,
                    vector_weight,
                    bm25_weight,
                    metadata_filters
                )
                
                if results:
                    # Store search results in session state
                    st.session_state.last_search_results = results
                    st.session_state.last_search_method = search_method
                    st.session_state.last_search_query = search_query
                    st.session_state.last_metadata_filters = metadata_filters
                    
                    print(f"DEBUG: About to display results with {len(st.session_state.custom_columns)} custom columns")
                    for col in st.session_state.custom_columns:
                        print(f"  - {col['display_name']}: {col['path']}")
                    
                    display_search_results(
                        results,
                        search_method,
                        show_metadata,
                        show_scores,
                        st.session_state.custom_columns
                    )
                else:
                    st.info("No results found. Try adjusting your search terms or thresholds.")
                    
            except Exception as e:
                print(f"ERROR: Search failed: {str(e)}")
                st.error(f"Search error: {str(e)}")
    
    # Display last search results if available and no new search
    elif st.session_state.last_search_results is not None:
        st.info(f"Showing results from last search: '{st.session_state.last_search_query}'")
        print(f"DEBUG: Displaying last search results with {len(st.session_state.custom_columns)} custom columns")
        display_search_results(
            st.session_state.last_search_results,
            st.session_state.last_search_method,
            show_metadata,
            show_scores,
            st.session_state.custom_columns
        )
    


def perform_search(
    rag_service: RAGService,
    project_id: int,
    query: str,
    method: str,
    page_size: int,
    similarity_threshold: float,
    rank_threshold: float,
    vector_weight: float,
    bm25_weight: float,
    metadata_filter: Optional[Dict] = None
) -> Optional[List[Dict]]:
    """Perform search based on selected method.
    
    Returns:
        List of result dictionaries with chunk data and scores
    """
    start_time = time.time()
    print(f"SEARCH: {method} - '{query}' (threshold: {similarity_threshold})")
    
    try:
        print(f"DEBUG: Metadata filter: {metadata_filter}")
        
        if method == "Semantic Search":
            results = rag_service.api.search_text(
                project_id=project_id,
                query_text=query,
                page=1,
                page_size=page_size,
                similarity_threshold=similarity_threshold,
                metadata_filter=metadata_filter
            )
        elif method == "BM25 (Keyword)":
            results = rag_service.api.search_bm25(
                project_id=project_id,
                query_text=query,
                page=1,
                page_size=page_size,
                rank_threshold=rank_threshold,
                metadata_filter=metadata_filter
            )
        elif method == "Hybrid Search":
            results = rag_service.api.search_hybrid(
                project_id=project_id,
                query_text=query,
                page=1,
                page_size=page_size,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                similarity_threshold=similarity_threshold,
                rank_threshold=rank_threshold,
                metadata_filter=metadata_filter
            )
        else:  # Metadata Query
            # For metadata queries, query_text is optional but metadata_filter is required
            if not metadata_filter:
                raise ValueError("Metadata filter is required for Metadata Query mode")
            
            results = rag_service.query_metadata(
                project_id=project_id,
                metadata_filter=metadata_filter,
                query_text=query if query and query.strip() else None,
                top_k=page_size
            )
        
        if not results or not hasattr(results, 'results') or not results.results:
            print(f"DEBUG: No results found")
            return None
        
        processing_start = time.time()
        result_count = len(results.results)
        print(f"PROCESSING: {result_count} results...")
        
        # Convert results to list of dictionaries
        formatted_results = []
        
        # Debug first result structure (only once)
        if results.results and result_count > 0:
            first_chunk = results.results[0].chunk
            if hasattr(first_chunk, 'metadata') and first_chunk.metadata:
                metadata_keys = list(first_chunk.metadata.keys())
                print(f"METADATA: {metadata_keys[:5]}{'...' if len(metadata_keys) > 5 else ''}")
                if 'filename' in first_chunk.metadata:
                    print(f"FILENAME: {first_chunk.metadata['filename']}")
                else:
                    print(f"FILENAME: Not found in metadata")
        
        for i, result in enumerate(results.results):
            try:
                chunk = result.chunk
                
                # Handle different chunk attribute names based on vector-rag structure
                # From the vector-rag Chunk model: content, index, metadata
                chunk_id = getattr(chunk, 'id', i)  # Fallback to array index
                chunk_index = getattr(chunk, 'index', i)
                content = getattr(chunk, 'content', str(chunk))
                metadata = getattr(chunk, 'metadata', {}) or {}
                
                # Get file information from metadata (same as chat functionality)
                # Try multiple possible keys for filename
                file_name = metadata.get('filename', metadata.get('file_name', metadata.get('title', 'Unknown')))
                file_id = metadata.get('file_id', None)
                
                # Try additional fallbacks if still unknown
                if file_name == 'Unknown':
                    if hasattr(chunk, 'file_name'):
                        file_name = chunk.file_name
                    elif hasattr(chunk, 'filename'):
                        file_name = chunk.filename
                
                if file_id is None:
                    if hasattr(chunk, 'file_id'):
                        file_id = chunk.file_id
                
                formatted_result = {
                    "id": chunk_id,
                    "content": content,
                    "file_name": file_name,
                    "file_id": file_id,
                    "chunk_index": chunk_index,
                    "score": result.score,
                    "metadata": metadata
                }
                
                # Add method-specific scores if available
                if method == "Hybrid Search" and "_scores" in metadata:
                    formatted_result["vector_score"] = metadata["_scores"].get("vector", 0)
                    formatted_result["bm25_score"] = metadata["_scores"].get("bm25", 0)
                
                formatted_results.append(formatted_result)
                
            except Exception as result_error:
                print(f"ERROR: Failed to process result {i+1}: {str(result_error)}")
                continue
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        print(f"DEBUG: Processed {len(formatted_results)} results in {processing_time:.2f}s")
        print(f"DEBUG: Total search time: {total_time:.2f}s")
        return formatted_results
        
    except Exception as e:
        print(f"ERROR: Exception in perform_search: {str(e)}")
        print(f"ERROR: Exception type: {type(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        st.error(f"Search error: {str(e)}")
        return None


def display_search_results(
    results: List[Dict],
    search_method: str,
    show_metadata: bool,
    show_scores: bool,
    custom_columns: List[Dict] = None
):
    """Display search results in a tabular format with expandable details."""
    
    # Use session state if custom_columns not provided
    if custom_columns is None:
        custom_columns = st.session_state.get('custom_columns', [])
    
    st.subheader(f"Search Results ({len(results)} found)")
    
    # Show active filters if any
    if st.session_state.get('last_metadata_filters'):
        with st.expander("🔍 Active Filters", expanded=False):
            filters = st.session_state.last_metadata_filters
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle comparison operators
                    for op, val in value.items():
                        op_symbol = {'$gt': '>', '$lt': '<', '$gte': '>=', '$lte': '<='}
                        st.text(f"{key} {op_symbol.get(op, op)} {val}")
                else:
                    st.text(f"{key} = {value}")
    
    # Debug custom columns
    print(f"DEBUG: Custom columns passed to display: {custom_columns}")
    print(f"DEBUG: Session state custom_columns: {st.session_state.get('custom_columns', [])}")
    
    # Column management UI
    with st.expander("🔧 Manage Columns", expanded=False):
        # Info message
        if st.session_state.custom_columns:
            st.info(f"ℹ️ {len(st.session_state.custom_columns)} custom columns configured. Click 'Search' button above to refresh the table with new columns.")
        
        # Quick presets
        st.markdown("**Quick Add Common Fields:**")
        preset_cols = st.columns(5)
        # Use metadata keys that are actually available based on debug output
        presets = [
            ("pages", "Pages"),
            ("type", "Type"),
            ("size", "Size"),
            ("chunker", "Chunker"),
            ("title", "Title")
        ]
        
        for i, (path, name) in enumerate(presets):
            with preset_cols[i % 5]:
                if st.button(f"+ {name}", key=f"preset_{path}"):
                    print(f"DEBUG: Preset button clicked - adding column: {name} ({path})")
                    new_column = {'path': path, 'display_name': name}
                    if new_column not in st.session_state.custom_columns:
                        st.session_state.custom_columns.append(new_column)
                        print(f"DEBUG: Added column to session state: {st.session_state.custom_columns}")
                        st.success(f"Added '{name}' column. Click 'Search' to update the table.")
                        # Force a rerun to update the display
                        st.rerun()
                    else:
                        st.warning(f"Column '{name}' already exists")
        
        st.markdown("---")
        
        # Custom column input
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            new_column_path = st.text_input(
                "Add custom metadata column",
                placeholder="e.g., extracted_fields[\"Drug Substance\"]",
                help="Enter a metadata path using dot notation or brackets for spaces"
            )
        
        with col2:
            custom_name = st.text_input(
                "Display name",
                placeholder="Optional",
                help="Leave empty to auto-generate"
            )
        
        with col3:
            if st.button("➕ Add Column", disabled=not new_column_path):
                print(f"DEBUG: Custom add button clicked - path: {new_column_path}")
                # Use custom name or parse from path
                if custom_name:
                    display_name = custom_name
                else:
                    display_name = new_column_path.split('.')[-1].replace('[', '').replace(']', '').replace('"', '').replace("'", '')
                
                new_column = {
                    'path': new_column_path,
                    'display_name': display_name
                }
                print(f"DEBUG: Creating new column: {new_column}")
                
                if new_column not in st.session_state.custom_columns:
                    st.session_state.custom_columns.append(new_column)
                    print(f"DEBUG: Added to session state. New custom_columns: {st.session_state.custom_columns}")
                    st.success(f"Added '{display_name}' column. Click 'Search' to update the table.")
                    # Force a rerun to update the display
                    st.rerun()
                else:
                    st.warning(f"Column '{display_name}' already exists")
        
        # Show current custom columns
        if st.session_state.custom_columns:
            st.markdown("**Current custom columns:**")
            for i, col in enumerate(st.session_state.custom_columns):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{col['display_name']} ({col['path']})")
                with col2:
                    if st.button(f"❌", key=f"remove_col_{i}"):
                        removed = st.session_state.custom_columns.pop(i)
                        print(f"DEBUG: Removed column: {removed}")
                        st.info(f"Removed '{removed['display_name']}' column")
                        st.rerun()
    
    # Create DataFrame for tabular display
    df_data = []
    for i, result in enumerate(results):
        row = {
            "Rank": i + 1,
            "File": result["file_name"],
            "Chunk": f"#{result['chunk_index']}",
            "Preview": (result["content"][:100] + "...") if len(result["content"]) > 100 else result["content"]
        }
        
        if show_scores:
            row["Score"] = f"{result['score']:.3f}"
            if search_method == "Hybrid Search":
                row["Vector"] = f"{result.get('vector_score', 0):.3f}"
                row["BM25"] = f"{result.get('bm25_score', 0):.3f}"
        
        # Add custom metadata columns
        print(f"DEBUG: Processing row {i+1} with {len(custom_columns)} custom columns")
        for col_def in custom_columns:
            metadata = result.get('metadata', {})
            value = extract_nested_value(metadata, col_def['path'])
            
            # Debug output
            if i == 0:  # Only debug first row
                print(f"DEBUG: Custom column '{col_def['display_name']}' path='{col_def['path']}'")
                print(f"DEBUG: Metadata keys: {list(metadata.keys())[:10]}")
                print(f"DEBUG: Metadata content sample: {str(metadata)[:200]}")
                print(f"DEBUG: Extracted value: {value}")
                
                # Try simple direct access for debugging
                if '.' not in col_def['path'] and '[' not in col_def['path']:
                    direct_value = metadata.get(col_def['path'])
                    print(f"DEBUG: Direct access value: {direct_value}")
                
                # Test the extract function
                test_paths = ['filename', 'size', 'chunker', 'embedder']
                for test_path in test_paths:
                    test_val = extract_nested_value(metadata, test_path)
                    if test_val is not None:
                        print(f"DEBUG: Test extract '{test_path}' = {test_val}")
            
            # Format the value for display
            if value is None:
                row[col_def['display_name']] = ""
            elif isinstance(value, (dict, list)):
                row[col_def['display_name']] = json.dumps(value)[:50] + "..."
            elif isinstance(value, float):
                row[col_def['display_name']] = f"{value:.3f}"
            elif isinstance(value, int):
                row[col_def['display_name']] = str(value)
            else:
                row[col_def['display_name']] = str(value)[:50]
        
        df_data.append(row)
    
    # Display as interactive table
    df = pd.DataFrame(df_data)
    
    # Debug DataFrame columns
    print(f"DEBUG: DataFrame columns: {list(df.columns)}")
    if df_data:
        print(f"DEBUG: First row keys: {list(df_data[0].keys())}")
        # Show custom column values if present
        for col_def in custom_columns:
            if col_def['display_name'] in df_data[0]:
                print(f"DEBUG: Custom column '{col_def['display_name']}' value: {df_data[0][col_def['display_name']]}")
    
    # Use container for better layout
    with st.container():
        # Build column configuration with optimized widths
        # Adjust Preview width based on number of columns
        total_columns = 4 + (1 if show_scores else 0) + (2 if search_method == "Hybrid Search" and show_scores else 0) + len(custom_columns)
        preview_width = "large" if total_columns <= 6 else "medium"  # type: ignore
        
        column_config = {
            "Rank": st.column_config.NumberColumn(width="small"),
            "File": st.column_config.TextColumn(width="medium"),
            "Chunk": st.column_config.TextColumn(width="small"),
            "Preview": st.column_config.TextColumn(width=preview_width),
        }
        
        if show_scores:
            column_config["Score"] = st.column_config.NumberColumn(width="small", format="%.3f")
            if search_method == "Hybrid Search":
                column_config["Vector"] = st.column_config.NumberColumn(width="small", format="%.3f")
                column_config["BM25"] = st.column_config.NumberColumn(width="small", format="%.3f")
        
        # Add configuration for custom columns
        for col_def in custom_columns:
            column_config[col_def['display_name']] = st.column_config.TextColumn(width="medium")
        
        # Display the dataframe
        # Debug the dataframe columns before display
        print(f"DEBUG: DataFrame has {len(df.columns)} columns: {list(df.columns)}")
        print(f"DEBUG: Column config has {len(column_config)} entries: {list(column_config.keys())}")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
    
    # Expandable details for each result
    st.markdown("### Result Details")
    
    for i, result in enumerate(results):
        with st.expander(f"🔍 Result {i + 1}: {result['file_name']} - Chunk #{result['chunk_index']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("**Full Content:**")
                st.text_area(
                    "Content",
                    value=result["content"],
                    height=200,
                    disabled=True,
                    key=f"content_{i}"
                )
            
            with col2:
                if show_scores:
                    st.markdown("**Scores:**")
                    st.metric("Overall Score", f"{result['score']:.3f}")
                    
                    if search_method == "Hybrid Search":
                        st.metric("Vector Score", f"{result.get('vector_score', 0):.3f}")
                        st.metric("BM25 Score", f"{result.get('bm25_score', 0):.3f}")
            
            if show_metadata and result.get("metadata"):
                st.markdown("**Metadata:**")
                display_metadata_tree(result["metadata"])
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📋 Copy to Clipboard", key=f"copy_{i}"):
                    # Note: Direct clipboard access requires JavaScript
                    st.info("Content copied! (Note: Use Ctrl+A, Ctrl+C to copy from text area above)")
            
            with col2:
                if st.button(f"💬 Send to Chat", key=f"chat_{i}"):
                    # Store in session state for chat to pick up
                    if "selected_chunks" not in st.session_state:
                        st.session_state.selected_chunks = []
                    st.session_state.selected_chunks.append(result)
                    st.success("Added to chat context!")
            
            with col3:
                if st.button(f"📄 View Full Document", key=f"doc_{i}"):
                    st.session_state.view_file_id = result["file_id"]
                    st.info("Switch to Files tab to view full document")


def display_metadata_tree(metadata: Dict):
    """Display metadata in a flat structure to avoid nested expanders."""
    
    def flatten_metadata(data: Dict, prefix: str = "") -> List[tuple]:
        """Flatten nested metadata into a list of (key_path, value) tuples."""
        items = []
        for key, value in data.items():
            # Skip internal keys
            if key.startswith("_"):
                continue
                
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                items.extend(flatten_metadata(value, full_key))
            elif isinstance(value, list):
                # Handle lists
                if value:  # Only show non-empty lists
                    list_items = []
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            list_items.append(f"[{i}] {json.dumps(item, indent=2)}")
                        else:
                            list_items.append(f"[{i}] {item}")
                    items.append((full_key, "\n".join(list_items)))
                else:
                    items.append((full_key, "[]"))
            else:
                # Simple value
                items.append((full_key, str(value)))
        
        return items
    
    # Flatten the metadata and display as formatted text
    flattened = flatten_metadata(metadata)
    
    if flattened:
        # Create a formatted string for all metadata
        metadata_text = []
        for key_path, value in flattened:
            # Add indentation based on nesting level
            indent_level = key_path.count('.')
            indent = "  " * indent_level
            
            # Format multi-line values
            if '\n' in str(value):
                metadata_text.append(f"{indent}**{key_path}:**")
                for line in str(value).split('\n'):
                    metadata_text.append(f"{indent}  {line}")
            else:
                metadata_text.append(f"{indent}**{key_path}:** {value}")
        
        # Display as a single code block for better formatting
        st.code("\n".join(metadata_text), language="yaml")
    else:
        st.info("No metadata available")


def render_search_statistics(rag_service: RAGService, project_id: int):
    """Render search statistics and analytics."""
    
    st.markdown("### 📊 Search Analytics")
    
    # Get project statistics
    try:
        files = rag_service.list_files(project_id)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", len(files))
        
        with col2:
            # This would need to be implemented in the RAG service
            st.metric("Total Chunks", "N/A")
        
        with col3:
            st.metric("Avg Chunks/File", "N/A")
        
        with col4:
            st.metric("Search Methods", "3")
        
    except Exception as e:
        st.error(f"Could not load statistics: {str(e)}")