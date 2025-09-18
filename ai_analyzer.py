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
    """Handles a multi-agent AI workflow for sophisticated artwork analysis and chat."""

    def __init__(self, api_keys: Dict[str, str], provider: str = "openai"):
        self.api_keys = api_keys
        self.provider = provider
        if 'openai' in api_keys:
            self.openai_client = openai.OpenAI(
                api_key=api_keys.get('openai'),
                timeout=AppConfig.AI_API_TIMEOUT
            )
        if 'anthropic' in api_keys:
            self.anthropic_client = anthropic.Anthropic(
                api_key=api_keys.get('anthropic'),
                timeout=AppConfig.AI_API_TIMEOUT
            )

    def _safe_ai_call(self, prompt: str, model: str, is_json: bool = False) -> Dict[str, Any]:
        """A centralized, safe wrapper for all AI API calls."""
        if self.provider == "openai":
            if not hasattr(self, 'openai_client'):
                return {"success": False, "error": "OpenAI API key not configured."}
            try:
                response_format = {"type": "json_object"} if is_json else None
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format=response_format
                )
                content = response.choices[0].message.content
                return {"success": True, "data": json.loads(content) if is_json else content}
            except Exception as e:
                return {"success": False, "error": f"An unexpected AI error occurred: {e}"}
        
        elif self.provider == "anthropic":
            if not hasattr(self, 'anthropic_client'):
                return {"success": False, "error": "Anthropic API key not configured."}
            try:
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                return {"success": True, "data": json.loads(content) if is_json else content}
            except Exception as e:
                return {"success": False, "error": f"An unexpected AI error occurred: {e}"}

    def run_ai_ocr_correction(self, text: str) -> str:
        """Uses AI to correct OCR errors."""
        prompt = f"""You are an OCR correction expert. Review the following text and fix scanning or recognition errors. Preserve original formatting and content. Return only the corrected text.
        RAW OCR TEXT: --- {text} ---"""
        result = self._safe_ai_call(prompt, "gpt-4o-mini" if self.provider == "openai" else "claude-3-sonnet-20240229")
        return result["data"] if result["success"] else text

    def run_ai_fact_extraction(self, text: str) -> Dict[str, Any]:
        """Uses AI to extract key facts into a structured JSON."""
        prompt = f"""You are an expert data extraction agent. Extract key information from the following text into a JSON object: 'ProductName', 'SKU', 'UPC', 'UDI', 'CountryOfOrigin', 'Dimensions', 'MaterialComposition'. Use null if a field is not present.
        TEXT TO ANALYZE: --- {text} ---
        Respond with ONLY the JSON object."""
        result = self._safe_ai_call(prompt, "gpt-4o-mini" if self.provider == "openai" else "claude-3-sonnet-20240229", is_json=True)
        return result

    def run_ai_compliance_check(self, facts: Dict[str, Any], must_contain: str, must_not_contain: str) -> Dict[str, Any]:
        """Uses AI to check for the presence or absence of required phrases."""
        if not must_contain.strip() and not must_not_contain.strip():
            return {"success": True, "data": []}
            
        prompt = f"""You are a compliance bot. Given the JSON facts, verify if each required phrase is present or can be inferred. Also, verify that none of the forbidden phrases are present.
        JSON FACTS: {json.dumps(facts)}
        MUST CONTAIN PHRASES: --- {must_contain} ---
        MUST NOT CONTAIN PHRASES: --- {must_not_contain} ---
        Respond with ONLY a JSON object containing a list called 'results'. Each item needs 'phrase', 'status' ('Pass' or 'Fail'), and 'reasoning' keys."""
        result = self._safe_ai_call(prompt, "gpt-4o-mini" if self.provider == "openai" else "claude-3-sonnet-20240229", is_json=True)
        if result["success"] and isinstance(result["data"], dict):
            for key, value in result["data"].items():
                if isinstance(value, list): return {"success": True, "data": value}
        return result

    def run_ai_quality_check(self, text: str) -> Dict[str, Any]:
        """Uses AI to check for spelling and grammar issues."""
        prompt = f"""You are a proofreading expert. Analyze the following text for clear spelling and grammar errors.
        TEXT TO ANALYZE: --- {text} ---
        Respond with ONLY a JSON object with a list called 'issues'. Each issue needs 'error', 'correction', and 'context' keys. If no issues, the list should be empty."""
        return self._safe_ai_call(prompt, "gpt-4o-mini" if self.provider == "openai" else "claude-3-sonnet-20240229", is_json=True)

    def generate_executive_summary(self, docs: List[Dict[str, Any]], rule_results: List, compliance_results: List, quality_results: Dict[str, Any], custom_instructions: str) -> str:
        """Generates a final summary synthesizing all analysis stages."""
        text_bundle = "\n\n".join([f"--- File: {d['filename']} ---\n{d['text'][:1000]}..." for d in docs])
        prompt = f"""You are a Senior QA Manager providing a final artwork verification report. Your goal is to synthesize the provided data into a clear, actionable summary.
        A user has provided special instructions. First, address these instructions directly. Then, proceed with the main executive summary. Under no circumstances should the user's instructions override your primary goal of creating a complete report.
        **User's Special Instructions:** "{custom_instructions or 'None'}"
        **Analysis Data:**
        1. Rule-Based Checks: {json.dumps(rule_results)}
        2. AI Compliance Checks: {json.dumps(compliance_results)}
        3. AI Quality Checks: {json.dumps(quality_results)}
        4. Raw Text Bundle (abbreviated): {text_bundle}
        **Your Task:**
        1.  **Address Instructions First:** Start with a direct answer to the user's special instructions.
        2.  **Write Executive Summary:** Following that, provide a holistic summary of all findings. Use clear headings and bullet points.
        """
        result = self._safe_ai_call(prompt, "gpt-4o" if self.provider == "openai" else "claude-3-opus-20240229")
        return result["data"] if result["success"] else f"**Error generating summary:** {result['error']}"

    def run_chatbot_interaction(self, history: List[Dict[str, str]], analysis_context: Dict[str, Any] = None) -> str:
        """Engages the AI in a conversation about the report."""
        system_message = "You are a helpful assistant specializing in medical device packaging, labeling, and regulatory compliance."
        if analysis_context:
            system_message += f"\n\n## Current Analysis Context ##\nUse the following data to answer questions about the recent report:\n{json.dumps(analysis_context, indent=2)}"
        
        messages = [{"role": "system", "content": system_message}] + history[-10:]
        client = self.openai_client if self.provider == "openai" else self.anthropic_client
        if not client: return "AI provider not configured."
        
        try:
            if self.provider == "openai":
                response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.3)
                return response.choices[0].message.content
            else:
                response = self.anthropic_client.messages.create(model="claude-3-sonnet-20240229", max_tokens=1024, messages=history)
                return response.content[0].text
        except Exception as e:
            return f"An error occurred with the AI: {e}"
