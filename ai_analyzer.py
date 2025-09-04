import streamlit as st
import openai
import anthropic
import time
import json
from typing import List, Dict, Any

from config import AppConfig

def check_api_keys() -> Dict[str, str]:
    """Checks for API keys in Streamlit secrets."""
    keys = {}
    if hasattr(st, 'secrets'):
        for key in ['openai_api_key', 'OPENAI_API_KEY']:
            if key in st.secrets and st.secrets[key]: keys['openai'] = st.secrets[key]
        for key in ['anthropic_api_key', 'ANTHROPIC_API_KEY', 'claude_api_key']:
            if key in st.secrets and st.secrets[key]: keys['anthropic'] = st.secrets[key]
    return keys

class AIReviewer:
    """Handles all communication with the AI APIs, including a two-stage analysis."""

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.openai_client = openai.OpenAI(api_key=api_keys.get('openai')) if 'openai' in api_keys else None
        self.anthropic_client = anthropic.Anthropic(api_key=api_keys.get('anthropic')) if 'anthropic' in api_keys else None

    def run_ai_fact_extraction(self, text: str) -> Dict[str, Any]:
        """Uses an AI to correct OCR errors and extract key facts into a structured JSON."""
        if not self.openai_client: return {"error": "OpenAI API key not found."}
        
        prompt = f"""
        You are an expert data extraction OCR agent. The following is raw text extracted from a product packaging file. 
        Your task is to correct any OCR errors and extract the key information into a structured JSON object.
        Extract the following fields: 'ProductName', 'SKU', 'UPC', 'UDI', 'CountryOfOrigin', and 'Dimensions'.
        If a field is not present, the value should be null.
        
        TEXT TO ANALYZE:
        ---
        {text}
        ---
        
        Respond with ONLY the JSON object.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI Fact Extraction Failed: {e}"}

    def _create_batches(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Creates text batches from documents to respect API context limits."""
        # ... (implementation remains the same as previous version)
        batches, current_batch = [], ""
        for doc in docs:
            doc_text = f"--- File: {doc['filename']} ---\n{doc['text']}"
            if len(current_batch) + len(doc_text) > AppConfig.AI_BATCH_MAX_CHARS:
                if current_batch: batches.append(current_batch)
                current_batch = doc_text
            else:
                current_batch += "\n\n" + doc_text
        if current_batch: batches.append(current_batch)
        return batches

    def generate_summary(self, docs: List[Dict[str, Any]], custom_instructions: str) -> str:
        """Processes documents in batches and returns a single, synthesized summary."""
        if not self.anthropic_client or not self.openai_client: return "Missing API keys for AI analysis."

        batches = self._create_batches(docs)
        if not batches: return "No text was extracted from the documents to review."
        
        all_reviews = []
        for i, batch in enumerate(batches):
            prompt = f"""You are a QA specialist. Review the following artwork text. Check for consistency in Product Name, SKU, UPC, and UDI. Flag issues like missing 'Made in China' text. {custom_instructions}. Present findings as a bulleted list, citing the filename for each point.\n\n---DATA---\n{batch}\n---END DATA---"""
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307", max_tokens=2048, 
                    messages=[{"role": "user", "content": prompt}]
                )
                all_reviews.append(response.content[0].text)
            except Exception as e:
                all_reviews.append(f"Anthropic API Error in batch {i+1}: {e}")
            time.sleep(1)

        final_prompt = f"""You are a senior QA manager. Consolidate the following individual analysis reports into a single, final summary.
        - For each finding, you MUST cite the source filename.
        - Do not mention 'batches' or 'reports'.
        - Structure the response using markdown with clear headings.
        - Custom instructions to address: {custom_instructions or 'None'}
        
        ---REPORTS TO SYNTHESIZE---\n{"\n\n---\n\n".join(all_reviews)}\n---END REPORTS---
        
        Provide your final, consolidated review below.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": final_prompt}], temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error during final synthesis: {e}"
