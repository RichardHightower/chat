-- Add full-text search support with handling for oversized chunks
-- This script truncates content that's too large for tsvector

-- First, add the column without the GENERATED clause
ALTER TABLE chunks 
ADD COLUMN IF NOT EXISTS content_tsv tsvector;

-- Create a function to safely convert text to tsvector with truncation
CREATE OR REPLACE FUNCTION safe_to_tsvector(input_text text) 
RETURNS tsvector AS $$
DECLARE
    max_length INTEGER := 1000000; -- Leave some buffer (actual max is 1048575)
    truncated_text TEXT;
BEGIN
    -- If text is too long, truncate it
    IF LENGTH(input_text) > max_length THEN
        truncated_text := LEFT(input_text, max_length);
        -- Try to truncate at a word boundary
        truncated_text := REGEXP_REPLACE(truncated_text, '\s+\S*$', '');
        RETURN to_tsvector('english', truncated_text);
    ELSE
        RETURN to_tsvector('english', input_text);
    END IF;
EXCEPTION
    WHEN OTHERS THEN
        -- If any error occurs, return an empty tsvector
        RETURN to_tsvector('english', '');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Update existing chunks with the safe tsvector
UPDATE chunks 
SET content_tsv = safe_to_tsvector(content)
WHERE content_tsv IS NULL;

-- Create a trigger to automatically update content_tsv on insert/update
CREATE OR REPLACE FUNCTION update_content_tsv() 
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := safe_to_tsvector(NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS chunks_content_tsv_trigger ON chunks;

-- Create the trigger
CREATE TRIGGER chunks_content_tsv_trigger
BEFORE INSERT OR UPDATE OF content ON chunks
FOR EACH ROW
EXECUTE FUNCTION update_content_tsv();

-- Create GIN index for efficient full-text search
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv 
ON chunks USING GIN (content_tsv);

-- Show statistics about the chunks that were truncated
SELECT 
    COUNT(*) as total_chunks,
    COUNT(CASE WHEN LENGTH(content) > 1000000 THEN 1 END) as truncated_chunks,
    MAX(LENGTH(content)) as max_content_length
FROM chunks;

-- Analyze table to update statistics for query planner
ANALYZE chunks;