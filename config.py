class AppConfig:
    """Holds the configuration constants for the application."""
    APP_TITLE = "Artwork Verification Tool"
    PAGE_ICON = "✅"
    
    # Keywords for classifying documents
    SHARED_FILE_KEYWORDS = ['manual', 'qsg', 'quickstart', 'washtag', 'logo']
    DOC_TYPE_MAP = {
        'packaging_artwork': ['packaging', 'box'],
        'manual': ['manual', 'qsg', 'quickstart'],
        'washtag': ['washtag', 'wash tag', 'care'],
        'shipping_mark': ['shipping', 'mark', 'carton'],
        'qc_sheet': ['qc', 'quality', 'sheet', 'specs', '.csv', '.xlsx'],
        'logo_tag': ['tag']
    }
    
    # AI configuration
    AI_BATCH_MAX_CHARS = 15000  # Max characters per API call to avoid limits
