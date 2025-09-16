class AppConfig:
    """Holds the configuration constants for the application."""
    # NEW: Renamed the app and updated the icon for a fresh look.
    APP_TITLE = "VAVE: Vive Health Artwork Verification Engine"
    PAGE_ICON = "🚀"

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
