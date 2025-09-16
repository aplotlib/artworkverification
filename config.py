import streamlit as st

class AppConfig:
    """Holds the configuration constants for the application."""
    # CORRECTED: Renamed the app to reflect "Vive Health"
    APP_TITLE = "Vive Health Artwork Verification Co-pilot"
    PAGE_ICON = "🎨"

    # UPGRADE: Added a configurable max file size for uploads.
    # Set to 200MB to accommodate high-resolution artwork files.
    MAX_FILE_SIZE_MB = 200

    # This setting is crucial for allowing large file uploads in Streamlit.
    # It should be set in your Streamlit server config, but we define it here for clarity.
    # Note: Streamlit's config option is `server.maxUploadSize`
    SERVER_MAX_UPLOAD_SIZE = MAX_FILE_SIZE_MB

    # Keywords for classifying documents as "shared" across variants
    SHARED_FILE_KEYWORDS = ['manual', 'qsg', 'quickstart', 'washtag', 'logo', 'thank you', 'ty_card']

    # Expanded mapping for more accurate document type classification
    DOC_TYPE_MAP = {
        'packaging_artwork': ['packaging', 'box'],
        'manual': ['manual', 'qsg', 'quickstart'],
        'wash_tag': ['washtag', 'wash tag', 'care'],
        'shipping_mark': ['shipping', 'mark', 'carton'],
        'qc_sheet': ['qc', 'quality', 'sheet', 'specs', '.csv', '.xlsx'],
        'logo_tag': ['logo_tag', 'logo'],
        'inner_tag': ['inner_tag'],
        'thank_you_card': ['thank you', 'ty_card']
    }

    # AI configuration
    AI_BATCH_MAX_CHARS = 10000  # Max characters per API call to avoid limits
    AI_API_TIMEOUT = 45 # seconds
