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

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.openai_client = openai.OpenAI(api_key=api_keys.get('openai')) if 'openai' in api_keys else None

    def run_ai_fact_extraction(self, text: str) -> Dict[str, Any]:
        """Uses an AI to correct OCR errors and extract key facts into a structured JSON."""
        if not self.openai_client: return {"error": "OpenAI API key not found."}
        prompt = f"""You are an expert data extraction agent. From the following raw text, correct any OCR errors and extract the key information into a JSON object. Extract: 'ProductName', 'SKU', 'UPC', 'UDI', 'CountryOfOrigin', 'Dimensions', and 'MaterialComposition'. If a field is not present, use null. TEXT TO ANALYZE: --- {text} --- Respond with ONLY the JSON object."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                temperature=0, response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI Fact Extraction Failed: {e}"}

    def run_ai_compliance_check(self, facts: Dict[str, Any], required_text: str) -> List[Dict[str, str]]:
        """Uses AI to check for the presence of required phrases against the extracted facts."""
        if not self.openai_client or not required_text: return []
        prompt = f"""You are a compliance bot. Given the JSON facts extracted from a document, verify if each of the following required phrases is present or can be inferred. For each phrase, provide a 'status' ('Pass' or 'Fail') and brief 'reasoning'.
        JSON FACTS: {json.dumps(facts)}
        REQUIRED PHRASES: --- {required_text} ---
        Respond with ONLY a JSON list of objects, where each object has 'phrase', 'status', and 'reasoning' keys."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                temperature=0, response_format={"type": "json_object"}
            )
            response_data = json.loads(response.choices[0].message.content)
            
            if isinstance(response_data, dict):
                for key, value in response_data.items():
                    if isinstance(value, list):
                        return value
            elif isinstance(response_data, list):
                return response_data
            
            return [] 
        except Exception as e:
            return [{"phrase": "AI Compliance Check", "status": "Fail", "reasoning": str(e)}]

    def run_ai_quality_check(self, text: str) -> Dict[str, Any]:
        """Uses AI to check for spelling and grammar issues."""
        if not self.openai_client: return {"error": "OpenAI API key not found."}
        prompt = f"""You are a proofreading expert. Analyze the following text for spelling and grammar errors. 
        TEXT TO ANALYZE: --- {text} ---
        Respond with ONLY a JSON object containing a list called 'issues'. Each issue should have 'error', 'correction', and 'context' keys. If no issues are found, return an empty list."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
                temperature=0, response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"AI Quality Check Failed: {e}"}

    def generate_executive_summary(self, docs: List[Dict[str, Any]], rule_results: List, compliance_results: List, quality_results: Dict[str, Any], custom_instructions: str) -> str:
        """Generates a final summary synthesizing all previous analysis stages."""
        if not self.openai_client: return "OpenAI API key not found."
        
        text_bundle = "\n\n".join([f"--- File: {d['filename']} ({d['file_nature']}) ---\n{d['text'][:1000]}..." for d in docs])
        
        prompt = f"""You are a Senior QA Manager providing a final artwork verification report.
        
        **User's Special Instructions:** "{custom_instructions or 'None'}"

        **Analysis Data:**
        1. Rule-Based Checks: {json.dumps(rule_results)}
        2. AI Compliance Checks: {json.dumps(compliance_results)}
        3. AI Quality Checks: {json.dumps(quality_results)}
        4. Raw Text Bundle (abbreviated): {text_bundle}

        **Your Task:**
        1.  **Address Instructions First:** Start your response with a direct answer to the user's special instructions.
        2.  **Write Executive Summary:** Following that, provide a holistic summary of all findings. Synthesize the rule-based, AI compliance, and AI quality checks with your own review of the text. Use clear headings, bullet points, and bold text to highlight key information and discrepancies.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error during final summary: {e}"

    def run_chatbot_interaction(self, history: List[Dict[str, str]], analysis_context: Dict[str, Any] = None) -> str:
        """Engages the AI in a conversation about medical device packaging."""
        if not self.openai_client: return "OpenAI API key not found."
        
        system_message = """You are a helpful assistant specializing in medical device packaging, labeling, and regulatory compliance. Answer questions clearly and concisely. You can talk about FDA regulations, UDI requirements, ISO standards, materials, and design best practices."""

        if analysis_context:
            system_message += f"\n\n## Current Analysis Context ##\nYou are also being asked to answer questions about a recently completed artwork verification report. Use the following data to answer any questions about the report:\n{json.dumps(analysis_context, indent=2)}"

        system_prompt = {"role": "system", "content": system_message}
        
        # --- Security: Chat History Capping ---
        capped_history = history[-10:]
        messages = [system_prompt] + capped_history
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"An error occurred with the AI: {e}"
