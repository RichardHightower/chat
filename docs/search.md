# Search Feature User Guide

## Overview

The chat application provides a powerful search interface that allows you to search through your document collection using four different methods. Each search method is optimized for different use cases, and results can be seamlessly integrated into your chat conversations for deeper analysis.

## Search Methods

### 1. Semantic Search (Vector Similarity)

**What it does:** Finds content that is conceptually similar to your query, even if it doesn't contain the exact words you searched for.

**How it works:** 
- Converts your query into a numerical vector representation (embedding)
- Compares it against pre-computed embeddings of all document chunks
- Returns chunks with the highest cosine similarity scores

**Best for:**
- Natural language questions ("What are the side effects of this medication?")
- Finding related concepts and synonyms
- Discovering content when you don't know the exact terminology
- Exploratory searches

**Example queries:**
- "patient outcomes after treatment"
- "how does the system handle errors"
- "benefits of using cloud storage"

**Settings:**
- **Similarity Threshold**: Minimum score (0.0-1.0) for results. Higher = more relevant but fewer results

### 2. BM25 Search (Keyword/Lexical)

**What it does:** Finds content containing your exact search terms using traditional keyword matching.

**How it works:**
- Uses PostgreSQL's full-text search with BM25 ranking algorithm
- Considers term frequency and document length
- Prioritizes rare terms over common ones

**Best for:**
- Searching for specific technical terms, acronyms, or product names
- Finding exact phrases or code snippets
- When you know the precise terminology
- Regulatory or compliance searches requiring exact matches

**Example queries:**
- "FDA-2023-N-0001"
- "getUserAuthentication()"
- "Section 4.2.1"

**Settings:**
- **BM25 Rank Threshold**: Minimum rank score (0.0-1.0) for results

### 3. Hybrid Search (Combined)

**What it does:** Combines semantic and keyword search to get the best of both approaches.

**How it works:**
- Runs both semantic and BM25 searches simultaneously
- Combines scores using configurable weights
- Returns results that match either semantically or lexically

**Best for:**
- General-purpose searching
- When you want both exact matches and related content
- Technical documentation where both concepts and specific terms matter
- Getting comprehensive results

**Example queries:**
- "authentication API security" (finds both the API docs and security best practices)
- "React useState performance" (finds exact React code and performance concepts)

**Settings:**
- **Semantic Weight**: 0.0-1.0 (higher favors conceptual matches)
- **BM25 Weight**: Automatically set to 1.0 - Semantic Weight
- Both threshold settings from semantic and BM25 apply

### 4. Metadata Query (Direct Filtering)

**What it does:** Retrieves chunks based on their metadata properties without any text matching or scoring.

**How it works:**
- Directly queries the PostgreSQL JSONB metadata column
- Uses exact matching for metadata fields
- Returns all matching chunks in database order (no relevance ranking)
- Does not compute embeddings or text similarity

**Best for:**
- Loading specific document sections or structures
- Navigating document hierarchies
- Bulk operations on specific content types
- When you know exactly what metadata properties you need

**Example queries:**
```
type=section
type=paragraph, section=Introduction
section=4. Clinical Results
document=report.pdf, page_number>=10
author.department=Engineering
```

**Note:** Search query is optional for metadata queries; filters are required.

## Using Metadata Filters

All search methods (except pure Metadata Query) support optional metadata filtering to narrow results.

### Filter Syntax

**Basic syntax:** `key=value, key2=value2`

**Supported operators:**
- `=` : Exact match
- `>` : Greater than
- `<` : Less than
- `>=` : Greater than or equal
- `<=` : Less than or equal

### Filter Examples

```
# Exact matches
type=toc
status=approved
language=en

# Numeric comparisons
pages>10
confidence_score>=0.8
year=2023

# Multiple filters (AND logic)
type=section, status=approved
pages>5, language=en

# Nested properties
author.name=Dr. Smith
metadata.version=2.0
```

## Search Interface Features

### Quick Filters

Pre-defined filter buttons for common searches:
- **TOC Only**: `type=toc` - Table of contents entries
- **Sections Only**: `type=section` - Document sections
- **Paragraphs**: `type=paragraph` - Regular text content
- **Large Docs**: `pages>10` - Documents with more than 10 pages

### Custom Columns

Add dynamic columns to the results table based on metadata fields:

1. Click "🔧 Manage Columns"
2. Use preset buttons or enter custom metadata paths:
   - Simple: `pages`, `type`, `author`
   - Nested: `author.name`, `metadata.version`
   - Arrays: `tags[0]`, `categories[1]`
3. Columns update immediately and persist across searches

### Results Display

Results are shown in two formats:

1. **Table View**: 
   - Rank, File, Chunk #, Preview
   - Optional: Scores, custom metadata columns
   - Sortable and scrollable

2. **Detailed View**:
   - Expandable sections for each result
   - Full content display
   - Complete metadata tree
   - Action buttons

## Integration with Chat

### Sending Results to Chat Context

Each search result can be added to your chat conversation:

1. Find relevant chunks using any search method
2. Click "💬 Send to Chat" on individual results
3. Selected chunks are added to `st.session_state.selected_chunks`
4. Switch to the Chat tab
5. Your next message will automatically include the selected chunks as context

### How Context Enhancement Works

When you send chunks to chat:

1. **Automatic Context**: The chat system includes your selected chunks
2. **Relevant Extraction**: The AI extracts and focuses on parts relevant to your question
3. **Source Attribution**: Responses reference the specific documents and sections
4. **Multi-chunk Synthesis**: The AI can combine information from multiple chunks

### Example Workflow

1. **Search Phase**:
   ```
   Method: Hybrid Search
   Query: "clinical trial efficacy results"
   Filter: type=section, year=2023
   ```

2. **Select Results**:
   - Review search results
   - Send relevant sections to chat (e.g., "4.2 Efficacy Analysis", "5.1 Statistical Summary")

3. **Chat Analysis**:
   ```
   You: "What were the primary efficacy endpoints and did the trial meet them?"
   
   AI: Based on the clinical trial sections you provided:
   
   From Section 4.2 Efficacy Analysis:
   - Primary endpoint: 28-day mortality reduction
   - Result: 15.2% reduction (p=0.003), meeting statistical significance
   
   From Section 5.1 Statistical Summary:
   - All pre-specified endpoints were met
   - Secondary endpoints showed consistent benefits...
   ```

## Advanced Tips

### Optimizing Search Performance

1. **For Known Documents**: Use Metadata Query with specific filters
2. **For Exploration**: Start with Semantic Search, refine with filters
3. **For Technical Terms**: Use BM25 or Hybrid with higher BM25 weight
4. **For Comprehensive Results**: Use Hybrid with balanced weights

### Combining Search Strategies

1. **Document Navigation**:
   - First: Metadata Query with `type=heading` to see structure
   - Then: Load specific sections with `section=X`

2. **Research Workflow**:
   - Start: Semantic search for concepts
   - Refine: Add metadata filters
   - Deep dive: Send key chunks to chat

3. **Technical Documentation**:
   - Use BM25 for function/API names
   - Use Semantic for "how to" questions
   - Use Metadata Query for version-specific docs

### Search Method Decision Tree

```
Need specific document sections?
  └─ Yes: Metadata Query
  └─ No: Continue ↓

Know exact terms/phrases?
  └─ Yes: Searching for code/technical terms?
      └─ Yes: BM25 Search
      └─ No: Hybrid Search (balanced)
  └─ No: Semantic Search
```

## Limitations and Notes

1. **Metadata Query**: Returns all results (no pagination limit)
2. **Search Scores**: Only available for Semantic, BM25, and Hybrid methods
3. **Filter Syntax**: Currently supports AND logic only (all filters must match)
4. **Performance**: Large result sets may be slow; use filters to narrow results

## Best Practices

1. **Start Broad, Then Narrow**: Begin with simple queries, add filters as needed
2. **Use Multiple Methods**: Different methods excel at different tasks
3. **Leverage Metadata**: Well-structured metadata enables powerful filtering
4. **Context is Key**: Send multiple relevant chunks to chat for comprehensive analysis
5. **Save Searches**: Note successful query/filter combinations for future use

This search interface transforms your document collection into a queryable knowledge base, enabling both precise retrieval and exploratory discovery, with seamless integration into AI-powered analysis through the chat interface.