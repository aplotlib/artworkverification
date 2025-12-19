from openai import OpenAI
import base64
import time

class AIAnalyzer:
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def analyze_problem(self, image_parts, user_text, topic, level):
        """
        Sends the user query + optional image to OpenAI GPT-4o.
        """
        
        # 1. Prepare the System Context
        system_msg = f"""
        You are an expert Economics Tutor (Level: {level}). 
        Current Topic: {topic}.
        
        GUIDELINES:
        - Explain the logic step-by-step.
        - Use LaTeX for math (e.g., $E_d$).
        - If an image is provided, explicitly analyze the graph axes and curves.
        """

        # 2. Prepare the User Content (Text + Image)
        user_content = [
            {
                "type": "text", 
                "text": f"Please solve/explain this problem:\n{user_text}"
            }
        ]

        # If an image exists, encode it to base64 and attach it
        if image_parts:
            try:
                # image_parts[0]['data'] comes as bytes from file_processor
                img_bytes = image_parts[0]['data']
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                mime_type = image_parts[0].get('mime_type', 'image/png')
                
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
            except Exception as e:
                return f"Error processing image for OpenAI: {e}"

        # 3. Send Request
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=1500,
                temperature=0.3 # Lower temperature for more precise math/logic
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"OpenAI API Error: {str(e)}"
