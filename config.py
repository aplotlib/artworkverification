import streamlit as st

class AppConfig:
    """Holds the configuration constants for the application."""
    APP_TITLE = "Vive Health Artwork Verification Co-pilot"
    PAGE_ICON = "🎨"
    MAX_FILE_SIZE_MB = 200
    SERVER_MAX_UPLOAD_SIZE = MAX_FILE_SIZE_MB
    SHARED_FILE_KEYWORDS = ['manual', 'qsg', 'quickstart', 'washtag', 'logo', 'thank you', 'ty_card']
    DOC_TYPE_MAP = {
        'packaging_artwork': ['packaging', 'box', 'sticker'],
        'manual': ['manual', 'qsg', 'quickstart'],
        'wash_tag': ['washtag', 'wash tag', 'care'],
        'shipping_mark': ['shipping', 'mark', 'carton'],
        'qc_sheet': ['qc', 'quality', 'sheet', 'specs', '.csv', '.xlsx'],
        'logo_tag': ['logo_tag', 'logo'],
        'udi_label': ['udi'],
        'thank_you_card': ['thank you', 'ty_card']
    }
    AI_BATCH_MAX_CHARS = 10000
    AI_API_TIMEOUT = 45

    BRAND_GUIDE = {
        "fonts": {"main": ["Poppins"], "secondary": ["Montserrat", "Arial"]},
        "colors": {
            "brand_color": {"name": "Vive Teal", "hex": "#23b2be", "rgb": (35, 178, 190)},
            "complementary_colors": [
                {"name": "Vive Dark Blue", "hex": "#004366", "rgb": (0, 67, 102)},
                {"name": "Vive Orange", "hex": "#eb3300", "rgb": (235, 51, 0)},
                {"name": "Vive Yellow", "hex": "#f0b323", "rgb": (240, 179, 35)},
                {"name": "Vive Gray", "hex": "#777473", "rgb": (119, 116, 115)}
            ]
        },
        "color_tolerance": 20.0
    }

    # UPGRADE: Added a default checklist based on analysis of provided files.
    DEFAULT_CHECKLIST = """
# Content & Compliance
- Country of Origin ("Made in China") is present on packaging and product.
- Warranty information (e.g., "1 year warranty") is correct.
- Distributor contact information is present and correct.
- SKU and UPC on packaging match the spec sheet.
- UDI is present on the packaging and matches the UDI file.
- Proposition 65 warning is included if required.

# File Presence
- Final packaging artwork file is included.
- Quick Start Guide / Manual file is included.
- Shipping Mark file is included.
- UDI label file is included.
- Wash tag / Care instructions file is included.

# Brand & Formatting
- Product dimensions on packaging match the spec sheet.
- All fonts comply with the VIVE Brand Guide (Poppins, Montserrat, Arial).
- All colors comply with the VIVE Brand Guide color palette.
- Logos are used correctly according to the brand guide.
"""
