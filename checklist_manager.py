import pandas as pd
import streamlit as st

class ChecklistManager:
    def __init__(self):
        self.vive_rules = []
        self.coretech_rules = []

    def load_checklist(self, file_path, brand_name):
        """
        Parses the 'Artwork Checklist' CSV format.
        It looks for the specific rows that have boolean checks or required items.
        """
        try:
            # Load CSV, skipping initial metadata rows if necessary (detected from your file structure)
            df = pd.read_csv(file_path, skiprows=2) 
            
            # Normalize columns to find the 'Item' or 'Requirement' column
            # Based on your snippet, requirements are in the first column often.
            rules = []
            
            # We iterate through the dataframe to find checklist items
            # Logic: If a row has a value in Column A that looks like a requirement
            for index, row in df.iterrows():
                item = str(row.iloc[0]).strip()
                
                # Filter out empty rows, headers, or non-requirements
                if item and item.lower() not in ['nan', 'packaging', 'manual', 'label', 'checked with ai']:
                    # Clean up the text (remove leading dashes used for indentation)
                    clean_item = item.lstrip('- ').strip()
                    
                    # Categorize the rule
                    category = "General"
                    if "UDI" in clean_item or "QR" in clean_item:
                        category = "Compliance"
                    elif "Dimension" in clean_item or "fit" in clean_item:
                        category = "Physical Spec"
                    elif "Color" in clean_item or "Logo" in clean_item:
                        category = "Branding"
                        
                    rules.append({
                        "id": f"{brand_name}_{index}",
                        "requirement": clean_item,
                        "category": category,
                        "original_text": item
                    })
            
            return rules
            
        except Exception as e:
            st.error(f"Error loading checklist for {brand_name}: {e}")
            return []

    def get_common_errors(self, error_tracker_path):
        """
        Loads the historical 'Artwork Error Tracker' to create a 'Watchlist'
        """
        try:
            df = pd.read_csv(error_tracker_path)
            # We are interested in the 'Issue Description' and 'Issue Category'
            errors = df[['Issue Description', 'Issue Category']].dropna().to_dict('records')
            return errors
        except Exception as e:
            # Fail silently if tracker isn't found, it's an enhancement, not a blocker
            print(f"Warning: Could not load error tracker: {e}")
            return []
