-- Check for chunks that are too large for tsvector
-- PostgreSQL tsvector has a maximum size of 1048575 bytes

-- First, let's see how many chunks exceed the limit
SELECT COUNT(*) as total_chunks,
       COUNT(CASE WHEN LENGTH(content) > 1048575 THEN 1 END) as chunks_too_large,
       COUNT(CASE WHEN LENGTH(content) > 900000 THEN 1 END) as chunks_near_limit
FROM chunks;

-- Show the largest chunks with their details
SELECT 
    c.id,
    c.chunk_index,
    f.name as file_name,
    LENGTH(c.content) as content_length,
    LEFT(c.content, 100) as content_preview,
    c.chunk_metadata
FROM chunks c
JOIN files f ON c.file_id = f.id
WHERE LENGTH(c.content) > 900000
ORDER BY LENGTH(c.content) DESC
LIMIT 20;

-- Group by file to see which files have oversized chunks
SELECT 
    f.id as file_id,
    f.name as file_name,
    COUNT(*) as total_chunks,
    COUNT(CASE WHEN LENGTH(c.content) > 1048575 THEN 1 END) as oversized_chunks,
    MAX(LENGTH(c.content)) as max_chunk_size,
    AVG(LENGTH(c.content))::INTEGER as avg_chunk_size
FROM files f
JOIN chunks c ON f.id = c.file_id
GROUP BY f.id, f.name
HAVING COUNT(CASE WHEN LENGTH(c.content) > 1048575 THEN 1 END) > 0
ORDER BY oversized_chunks DESC;