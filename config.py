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
        "fonts": {
            "main": ["Poppins"],
            "secondary": ["Montserrat", "Arial"]
        },
        "colors": {
            "brand_color": {"name": "Vive Teal", "hex": "#23b2be", "rgb": (35, 178, 190)},
            "complementary_colors": [
                {"name": "Vive Dark Blue", "hex": "#004366", "rgb": (0, 67, 102)},
                {"name": "Vive Orange", "hex": "#eb3300", "rgb": (235, 51, 0)},
                {"name": "Vive Yellow", "hex": "#f0b323", "rgb": (240, 179, 35)},
                {"name": "Vive Gray", "hex": "#777473", "rgb": (119, 116, 115)}
            ]
        },
        "color_tolerance": 20.0,
        "COLOR_IGNORE_LIST": [
            (0, 0, 0),       # Black
            (255, 255, 255)  # White
        ]
    }

    CHECKLISTS = {
        "Vive": {
            "Packaging": ["Product Name Consistency", "SKU ID", "UPC and UDI", "Image Visual", "Country Origin (Made in China)", "Made in China sticker"],
            "Manual": ["Product Name Consistency", "SKU ID", "Outlined Texts", "QR Code matches the shortlink", "Country Origin", "Check dims if it will fit in the box"],
            "Inserts/Stickers": ["Thank you Card (Vive Products only)"],
            "Quickstart, IFU": ["Product Name Consistency", "SKU ID", "Outlined Texts", "QR Code matches the shortlink", "UDI", "Product QR Code", "Master Carton UDI and Shipping Mark", "Giftbox UDI on Packaging"],
            "Washtag": ["Multivariant? (one washtag/size or color)", "Country Origin"],
            "Shipping Mark": ["Confirm qty-ctn from R&D", "Confirm Origin (Made in ...)"],
            "QC Sheet (Cross Check with R&D)": ["Packaging Dims", "Logo Print/tag placement/color", "Wash tag placement/color", "UDI Info", "Sticker placements", "Match barcode on artworks"]
        },
        "Coretech": {
            "Packaging": ["Product Name Consistency", "SKU ID", "UPC and UDI", "Image Visual", "Country Origin (Made in China)"],
            "Billing Sticker": ["SKU ID", "UPC and UDI", "HCPCS and Lot #", "Match Size to Barcode and UDI"],
            "Manual": ["Product Name Consistency", "SKU ID", "Outlined Texts", "QR Code matches the shortlink", "Country Origin", "Check dims if it will fit in the box"],
            "Other Inserts/Sticker": ["Vive Now, Neoprene Insert, Air out insert, Fit info, etc.", "Warning label"],
            "Quickstart, IFU": ["Product Name Consistency", "SKU ID", "Outlined Texts", "QR Code matches the shortlink", "UDI", "Product QR Code", "Master Carton UDI an Shipping Mark"],
            "Washtag": ["Multivariant? (one washtag/size or color)", "Country Origin"],
            "Shipping Mark": ["Confirm qty-ctn from R&D", "Confirm Origin (Made in ...)"],
            "QC Sheet (Cross Check with R&D)": ["Packaging Dims", "Logo Print/tag placement/color", "Wash tag placement/color", "UDI Info", "Sticker placements", "Match barcode on artworks"]
        }
    }
