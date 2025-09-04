import streamlit as st
import openai
import anthropic
import time
from typing import List, Dict, Any
from config import AppConfig

def check_api_keys() -> Dict[str, str]:
    """Checks for API keys in Streamlit secrets."""
    keys = {}
    if hasattr(st, 'secrets'):
        for key in ['openai_api_key', 'OPENAI_API_KEY']:
            if key in st.secrets and st.secrets[key]:
                keys['openai'] = st.secrets[key]
        for key in ['anthropic_api_key', 'ANTHROPIC_API_KEY', 'claude_api_key']:
            if key in st.secrets and st.secrets[key]:
                keys['anthropic'] = st.secrets[key]
    return keys

class AIReviewer:
    """Handles all communication with the AI APIs, including batching and chained reviews."""

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.openai_client = openai.OpenAI(api_key=api_keys.get('openai')) if 'openai' in api_keys else None
        self.anthropic_client = anthropic.Anthropic(api_key=api_keys.get('anthropic')) if 'anthropic' in api_keys else None

    def _create_batches(self, docs: List[Dict[str, Any]]) -> List[str]:
        """Creates text batches from documents to respect API context limits."""
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

    def _get_batch_review(self, text_bundle: str, custom_instructions: str) -> str:
        if not self.anthropic_client: return "Anthropic API key not found."
        prompt = f"You are a QA specialist. Review the following artwork text from one or more files. Check for consistency in Product Name, SKU, UPC, and UDI. Flag issues like missing 'Made in China' text. {custom_instructions}. Present findings as a bulleted list, citing the filename for each point.\n\n---DATA---\n{text_bundle}\n---END DATA---"
        try:
            response = self.anthropic_client.messages.create(model="claude-3-haiku-20240307", max_tokens=2048, messages=[{"role": "user", "content": prompt}])
            return response.content[0].text
        except Exception as e: return f"Anthropic API Error: {e}"

    def _get_final_synthesis(self, all_batch_reviews: str, custom_instructions: str) -> str:
        if not self.openai_client: return "OpenAI API key not found."
        prompt = f"""You are a senior QA manager. Consolidate the following individual analysis reports into a single, final summary.
        - For each finding, you MUST cite the source filename it came from.
        - Do not mention 'batches' or 'reports'.
        - Structure the response using markdown with clear headings for Key Findings, Discrepancies, and Recommendations.
        - If custom instructions were provided, ensure they are addressed in the summary. Custom instructions: {custom_instructions}
        
        ---REPORTS TO SYNTHESIZE---\n{all_batch_reviews}\n---END REPORTS---
        
        Provide your final, consolidated review below.
        """
        try:
            response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI API Error: {e}"

    def generate_summary(self, docs: List[Dict[str, Any]], custom_instructions: str) -> str:
        """Processes documents in batches and returns a single, synthesized summary."""
        batches = self._create_batches(docs)
        if not batches: return "No text was extracted from the documents to review."
        
        all_reviews = []
        progress_bar = st.progress(0, text="Starting AI review...")

        for i, batch in enumerate(batches):
            progress_text = f"AI Review: Processing Batch {i+1} of {len(batches)}..."
            progress_bar.progress((i + 0.5) / len(batches), text=progress_text)
            
            cust_instr = f"Pay special attention to: '{custom_instructions}'" if custom_instructions else ""
            # Using Anthropic for initial, fast review of batches
            summary = self._get_batch_review(batch, cust_instr)
            all_reviews.append(summary)
            time.sleep(1) # Crucial delay to prevent rate limiting

        progress_bar.progress(1.0, text="Consolidating AI analysis...")
        
        # Using OpenAI for a final, high-quality synthesis
        final_summary = self._get_final_synthesis("\n\n---\n\n".join(all_reviews), cust_instr)
        
        progress_bar.empty()
        return final_summary
