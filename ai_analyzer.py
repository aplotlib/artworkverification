from openai import OpenAI
import base64

class AIAnalyzer:
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def analyze_problem(self, image_parts, user_text, topic, level):
        """
        Solves specific homework problems.
        """
        system_msg = f"""
        You are an expert Economics Tutor (Level: {level}). Topic: {topic}.
        1. Explain logic step-by-step.
        2. Use LaTeX for math ($...$).
        3. If variables are provided (e.g. P=10, Q=5), solve for the missing variable using standard economic formulas.
        """
        
        content_payload = [{"type": "text", "text": user_text}]

        if image_parts:
            try:
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

    def generate_study_guide(self, topic, level):
        """
        Generates a structured study guide for a specific topic.
        """
        prompt = f"""
        Create a comprehensive Economics Study Guide for the topic: "{topic}" (Level: {level}).
        
        Structure the guide exactly like this:
        1. **Key Definitions**: Brief, clear definitions of core terms.
        2. **Core Formulas**: List the math formulas with variable definitions.
        3. **Concept Explanation**: Explain the "Intuition" behind the concept.
        4. **Common Pitfalls**: What do students usually get wrong?
        5. **Practice Question**: One example problem with a step-by-step solution.
        
        Format using Markdown.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful academic assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating guide: {str(e)}"
