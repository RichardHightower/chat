import json
import re
from pathlib import Path
from typing import Optional

from chat.util.logging_util import logger


class JsonUtil:
    """Utility class for handling JSON operations."""

    @staticmethod
    def extract_json(raw_json_str: str) -> dict:
        """
        Extract and parse JSON from a string that might contain additional text.

        Args:
            raw_json_str: The raw string that contains JSON
        Returns:
            Parsed JSON as a dictionary

        Raises:
            json.JSONDecodeError: If JSON parsing fails
        """
        try:
            # First try: direct parsing if it's already valid JSON
            try:
                return json.loads(raw_json_str)
            except json.JSONDecodeError:
                pass # Continue to next method if this fails

            # Try removing markdown code block syntax
            if raw_json_str.startswith('```json') and raw_json_str.endswith('```'):
                try:
                    # Slice off ```json and ```
                    return json.loads(raw_json_str[7:-3].strip())
                except json.JSONDecodeError:
                    pass # Continue
            elif raw_json_str.startswith('```') and raw_json_str.endswith('```'):
                try:
                    # Slice off ``` and ```
                    return json.loads(raw_json_str[3:-3].strip())
                except json.JSONDecodeError:
                    pass # Continue


            # Second try: Extract JSON from code blocks using regex
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            code_blocks = re.findall(code_block_pattern, raw_json_str)

            if code_blocks:
                for block in code_blocks:
                    try:
                        return json.loads(block.strip())
                    except json.JSONDecodeError:
                        continue # Try next block if current one fails

            # Third try: Special case for responses that start with ```json but might not have closing backticks
            stripped_raw_str = raw_json_str.strip()
            if stripped_raw_str.startswith("```json") or stripped_raw_str.startswith("```"):
                # Extract everything after the opening backticks
                json_content = re.sub(r"^```(?:json)?\s*", "", stripped_raw_str)
                # Remove closing backticks if they exist at the very end of the content
                json_content = re.sub(r"\s*```\s*$", "", json_content)
                try:
                    return json.loads(json_content.strip())
                except json.JSONDecodeError:
                    logger.debug("Failed to parse JSON from code block with potentially incomplete delimiters")


            # Fourth try: Look for JSON object delimiters {} or array delimiters []
            # Try to find the first '{' and last '}' to extract a JSON object
            start_idx_obj = raw_json_str.find('{')
            end_idx_obj = raw_json_str.rfind('}')

            if start_idx_obj != -1 and end_idx_obj != -1 and end_idx_obj > start_idx_obj:
                potential_json_obj = raw_json_str[start_idx_obj : end_idx_obj + 1]
                try:
                    # Validate by trying to parse this substring
                    # More robustly find the actual end of this object
                    brace_count = 0
                    final_end_idx = -1
                    for i, char in enumerate(raw_json_str[start_idx_obj:], start=start_idx_obj):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                final_end_idx = i
                                break
                    if final_end_idx != -1:
                        json_candidate = raw_json_str[start_idx_obj : final_end_idx + 1]
                        return json.loads(json_candidate)
                except json.JSONDecodeError:
                    pass # Object parsing failed, try array or give up

            # Fifth try: Look for JSON array delimiters []
            start_idx_arr = raw_json_str.find('[')
            end_idx_arr = raw_json_str.rfind(']')

            if start_idx_arr != -1 and end_idx_arr != -1 and end_idx_arr > start_idx_arr:
                potential_json_arr = raw_json_str[start_idx_arr : end_idx_arr + 1]
                try:
                     # Validate by trying to parse this substring (similar logic for arrays if needed)
                    bracket_count = 0
                    final_end_idx_arr = -1
                    for i, char in enumerate(raw_json_str[start_idx_arr:], start=start_idx_arr):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                final_end_idx_arr = i
                                break
                    if final_end_idx_arr != -1:
                        json_candidate_arr = raw_json_str[start_idx_arr : final_end_idx_arr + 1]
                        return json.loads(json_candidate_arr)
                except json.JSONDecodeError:
                    pass # Array parsing failed

            # If we got here, all parsing attempts failed
            logger.error(f"Could not find valid JSON in the response: {raw_json_str[:500]}...")
            raise json.JSONDecodeError("Could not find valid JSON in the response", raw_json_str, 0)

        except Exception as e: # Catch any other unexpected error during extraction
            logger.error(f"Unexpected error during JSON extraction: {e}", exc_info=True)
            logger.error(f"Raw response snippet: {raw_json_str[:500]}...")
            raise # Re-raise the original or a new error

    @staticmethod
    def extract_and_parse_json(raw_json_str: str, debug_file_path: Optional[Path] = None,
                               debug_file_name: str = "raw_json_response.txt") -> dict:
        """
        Extract and parse JSON from a string that might contain additional text.
        Optionally saves the raw response for debugging.
        """
        if debug_file_path:
            # Ensure debug_file_path is a Path object if it's a string
            if isinstance(debug_file_path, str):
                debug_file_path = Path(debug_file_path)

            debug_file_path.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            raw_response_path = debug_file_path / debug_file_name
            try:
                with open(raw_response_path, "w", encoding="utf-8") as f:
                    f.write(raw_json_str)
                logger.info(f"Raw response saved to {raw_response_path}")
            except Exception as e:
                logger.error(f"Failed to save raw response to {raw_response_path}: {e}")

        return JsonUtil.extract_json(raw_json_str)

    @staticmethod
    def format_json_schema(schema: dict, indent: int = 2) -> str:
        """
        Format a JSON schema in a more readable way.
        """
        try:
            formatted_json = json.dumps(schema, indent=indent, sort_keys=False)
            logger.debug("Successfully formatted JSON schema")
            return formatted_json
        except Exception as e:
            logger.error(f"Failed to format JSON schema: {e}")
            return str(schema) # Fallback
