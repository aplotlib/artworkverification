import io
from PIL import Image
import fitz  # PyMuPDF

class FileProcessor:
    @staticmethod
    def process_file(uploaded_file):
        """
        Returns a tuple: (text_content, image_parts_for_gemini, preview_image)
        """
        filename = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        
        extracted_text = ""
        image_parts = []
        preview_img = None
        
        if filename.lower().endswith('.pdf'):
            # Open PDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Extract text from all pages
            for page in doc:
                extracted_text += page.get_text() + "\n"
            
            # Render first page as image for AI and Preview
            if len(doc) > 0:
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better quality
                img_data = pix.tobytes("png")
                preview_img = Image.open(io.BytesIO(img_data))
                
                image_parts = [
                    {
                        "mime_type": "image/png",
                        "data": img_data
                    }
                ]
                
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Image processing
            image = Image.open(uploaded_file)
            preview_img = image
            
            # Simple OCR could go here if needed, but we rely on Gemini for image text mostly
            extracted_text = "Image file - Text extraction relied on AI."
            
            image_parts = [
                {
                    "mime_type": uploaded_file.type,
                    "data": file_bytes
                }
            ]
            
        return extracted_text, image_parts, preview_img
