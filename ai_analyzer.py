from openai import OpenAI
import base64

class AIAnalyzer:
    def __init__(self, api_key, model_name):
        # Initialize the official OpenAI Client
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def analyze_problem(self, image_parts, user_text, topic, level):
        """
        Sends the user query + optional image to OpenAI.
        """
        
        system_msg = f"""
        You are an expert Economics Tutor (Level: {level}). Topic: {topic}.
        1. Explain logic step-by-step.
        2. Use LaTeX for math ($...$).
        3. If an image is present, analyze the graph/data specifically.
        """

        # Build Message
        content_payload = [{"type": "text", "text": user_text}]

        # Add Image if present
        if image_parts:
            try:
                # image_parts[0]['data'] is bytes. Convert to base64.
                b64_img = base64.b64encode(image_parts[0]['data']).decode('utf-8')
                mime = image_parts[0].get('mime_type', 'image/png')
                
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_img}"}
                })
            except Exception as e:
                return f"Error processing image: {e}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": content_payload}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {str(e)}"
