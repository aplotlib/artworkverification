import google.generativeai as genai
import json
import time

class AIAnalyzer:
    def __init__(self, api_key, model_name):
        if api_key:
            genai.configure(api_key=api_key)
        self.model_name = model_name

    def analyze_problem(self, image_parts, user_text, topic, level):
        """
        Sends the user query + optional image to Gemini with an Economics persona.
        """
        
        prompt = f"""
        ACT AS: Economics Tutor (Level: {level})
        TOPIC: {topic}
        
        USER QUESTION:
        {user_text}
        
        TASK:
        1. Identify the core economic concept (e.g., Opportunity Cost, Nash Equilibrium, IS-LM).
        2. If an image is attached, extract the data/graph details.
        3. Solve the problem step-by-step.
        4. Provide the final answer clearly.
        5. List any relevant formulas used.
        """
        
        try:
            model = genai.GenerativeModel(self.model_name)
            
            # Prepare content parts: Prompt first, then image if it exists
            content = [prompt]
            if image_parts:
                content.append(image_parts[0]) # Processing the first image found
            
            # Retry logic
            for attempt in range(3):
                try:
                    # Stream = False for simple response
                    response = model.generate_content(content)
                    return response.text
                except Exception as e:
                    time.sleep(2 * (attempt + 1))
            
            return "Error: AI Service Unavailable. Please check your API Key or try again."

        except Exception as e:
            return f"System Error: {str(e)}"
