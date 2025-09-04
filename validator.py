import re
from typing import List, Tuple, Dict
from collections import defaultdict

class ArtworkValidator:
    """Performs core, rule-based validation checks on the combined text."""

    def __init__(self, all_text: str, reference_text: str = None):
        """
        Args:
            all_text: A single string containing all text from all documents.
            reference_text: Optional string of mandatory text to check for, with phrases separated by newlines.
        """
        self.all_text = all_text
        self.reference_text = reference_text

    def validate(self, docs: List[Dict[str, any]]) -> Tuple[List, Dict]:
        """Runs all validation checks and returns global and per-document results."""
        global_results = []
        per_doc_results = defaultdict(list)
        
        # --- Global Check: UDI & UPC Analysis ---
        cleaned_text = self.all_text.replace(" ", "").replace("\n", "")
        upcs = set(re.findall(r'(\d{12})', cleaned_text))
        udis = set(re.findall(r'\(01\)(\d{14})', cleaned_text))

        if not upcs and not udis:
            global_results.append(('failed', "No UPCs or UDIs found across all documents.", "no_serials"))
        else:
            if not upcs:
                global_results.append(('failed', "UDIs were found, but no matching 12-digit UPCs were detected.", "no_upcs"))
            for upc in upcs:
                if not any(upc in udi[2:] for udi in udis):
                    global_results.append(('failed', f"UPC `{upc}` does not have a matching UDI.", f"mismatch_{upc}"))
                else:
                    global_results.append(('passed', f"UPC `{upc}` has a matching UDI.", f"match_{upc}"))
        
        # --- Per-Document Check: Reference Text ---
        if self.reference_text:
            # Split user input into individual required phrases
            required_phrases = [phrase.strip() for phrase in self.reference_text.split('\n') if phrase.strip()]
            
            for doc in docs:
                doc_text_lower = doc['text'].lower()
                for phrase in required_phrases:
                    if phrase.lower() not in doc_text_lower:
                        per_doc_results[doc['filename']].append(('failed', f"Missing required text: '{phrase}'"))
                    else:
                        per_doc_results[doc['filename']].append(('passed', f"Found required text: '{phrase}'"))

        return global_results, per_doc_results
