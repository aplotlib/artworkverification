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
                batches.append(current_batch)
                current_batch = doc_text
            else:
                current_batch += "\n\n" + doc_text
        if current_batch:
            batches.append(current_batch)
        return batches

    def _get_anthropic_review(self, text_bundle: str, custom_instructions: str) -> str:
        if not self.anthropic_client: return "Anthropic API key not found."
        prompt = f"You are a QA specialist. Review the following artwork text. Check for consistency in Product Name, SKU, UPC, and UDI. Flag issues like missing 'Made in China' text. {custom_instructions}. Present findings as a bulleted list.\n\n---DATA---\n{text_bundle}\n---END DATA---"
        try:
            response = self.anthropic_client.messages.create(model="claude-3-haiku-20240307", max_tokens=2048, messages=[{"role": "user", "content": prompt}])
            return response.content[0].text
        except Exception as e: return f"Anthropic API Error: {e}"

    def _get_openai_synthesis(self, text_bundle: str, anthropic_review: str, custom_instructions: str) -> str:
        if not self.openai_client: return "OpenAI API key not found."
        prompt = f"You are a senior QA manager. Provide a final, consolidated summary based on an initial AI review and the source text. Correct any mistakes from the first review.\n{custom_instructions}\n---ORIGINAL TEXT---\n{text_bundle}\n---CLAUDE'S REVIEW---\n{anthropic_review}\n---END REVIEWS---\nProvide your final, synthesized review below as a bulleted list. Start with '### Final AI Analysis'."
        try:
            response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0)
            return response.choices[0].message.content
        except Exception as e: return f"OpenAI API Error: {e}"

    def generate_summary_in_batches(self, provider: str, docs: List[Dict[str, Any]], custom_instructions: str) -> str:
        """Processes documents in batches and returns a single summary string."""
        batches = self._create_batches(docs)
        all_summaries = []
        
        progress_bar = st.progress(0, text="Starting AI review...")

        for i, batch in enumerate(batches):
            progress_text = f"Processing Batch {i+1} of {len(batches)}..."
            progress_bar.progress((i + 1) / len(batches), text=progress_text)
            
            cust_instr = f"Pay special attention to: '{custom_instructions}'" if custom_instructions else ""
            summary = ""
            if provider == 'both':
                anthropic_review = self._get_anthropic_review(batch, cust_instr)
                summary = self._get_openai_synthesis(batch, anthropic_review, cust_instr)
            elif provider == 'anthropic':
                 summary = self._get_anthropic_review(batch, cust_instr)
            elif provider == 'openai':
                 summary = self._get_openai_synthesis(batch, "", cust_instr)
            
            all_summaries.append(f"### Review of Batch {i+1}\n" + summary)
            time.sleep(1) # Crucial delay to prevent rate limiting

        progress_bar.empty()
        return "\n\n---\n\n".join(all_summaries)
