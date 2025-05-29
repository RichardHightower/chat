-- Identify chunks that exceed the tsvector limit
-- Run this BEFORE applying the FTS migration to see what will be affected

-- Show exact chunks that exceed the limit
SELECT 
    c.id as chunk_id,
    c.file_id,
    f.name as file_name,
    c.chunk_index,
    LENGTH(c.content) as content_bytes,
    LENGTH(c.content) - 1048575 as bytes_over_limit,
    ROUND((LENGTH(c.content) / 1048575.0) * 100, 2) as percent_of_limit,
    LEFT(c.content, 200) || '...' as content_preview,
    c.chunk_metadata->>'chunk_strategy' as chunk_strategy
FROM chunks c
JOIN files f ON c.file_id = f.id  
WHERE LENGTH(c.content) > 1048575
ORDER BY LENGTH(c.content) DESC;

-- Summary by chunking strategy (if available in metadata)
SELECT 
    c.chunk_metadata->>'chunk_strategy' as chunk_strategy,
    COUNT(*) as total_chunks,
    COUNT(CASE WHEN LENGTH(c.content) > 1048575 THEN 1 END) as oversized_chunks,
    MAX(LENGTH(c.content)) as max_size,
    AVG(LENGTH(c.content))::INTEGER as avg_size
FROM chunks c
GROUP BY c.chunk_metadata->>'chunk_strategy'
ORDER BY oversized_chunks DESC;