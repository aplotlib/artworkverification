import google.generativeai as genai
import json
import time

class AIAnalyzer:
    def __init__(self, api_key, model_name):
        if not api_key:
            # Just a placeholder, real key comes from main
            pass
        self.model_name = model_name

    def analyze_artwork(self, image_parts, checklist_context, historical_errors):
        """
        Sends the image + checklist + error history to Gemini.
        """
        
        # Construct a context-rich prompt
        error_context = "\n".join([f"- {e['Issue Description']} (Category: {e['Issue Category']})" for e in historical_errors[:5]])
        
        prompt = f"""
        You are a QA Validator for medical packaging. Analyze the attached artwork image.
        
        Context:
        1. BRAND CHECKLIST: The following items MUST be present:
        {checklist_context}
        
        2. KNOWN HISTORICAL ERRORS (Be extra vigilant for these):
        {error_context}
        
        Perform these specific checks:
        1. **SKU & Variant Match**: Does the color of the product in the image match the text description/SKU? (e.g., if Text says 'Gray', Image must not be 'Black').
        2. **Dimensions**: If dimensions are listed (e.g., 25x25cm), do they seem visually plausible for the container?
        3. **Washtag/Icons**: Are standard laundry/care icons present and clear?
        4. **Spelling**: Check all large text for typos.
        
        Output format: JSON object with a list of 'findings'.
        Example:
        {{
            "findings": [
                {{"check": "SKU Color Match", "status": "PASS", "observation": "Product is purple, SKU matches purple."}},
                {{"check": "Made in China", "status": "FAIL", "observation": "Text not found on back panel."}}
            ]
        }}
        """
        
        try:
            # In a real app, you would initialize genai with the key in main.py or config
            # Here we assume the global instance or passed client is used.
            # For this file generation, we assume standard generation logic.
            
            model = genai.GenerativeModel(self.model_name)
            
            # Retry logic for robustness
            for attempt in range(3):
                try:
                    response = model.generate_content([prompt, image_parts[0]], generation_config={"response_mime_type": "application/json"})
                    return json.loads(response.text)
                except Exception as e:
                    time.sleep(2 * (attempt + 1))
            
            return {"findings": [{"check": "AI Analysis", "status": "ERROR", "observation": "AI Service Unavailable"}]}

        except Exception as e:
            return {"findings": [{"check": "AI Analysis", "status": "ERROR", "observation": str(e)}]}
