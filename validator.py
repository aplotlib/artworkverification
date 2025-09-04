import re
from typing import List, Tuple

class ArtworkValidator:
    """Performs core, rule-based validation checks on the combined text."""

    def __init__(self, all_text: str, reference_text: str = None):
        """
        Args:
            all_text: A single string containing all text from all documents.
            reference_text: Optional string of mandatory text to check for.
        """
        self.all_text = all_text
        self.reference_text = reference_text

    def validate(self) -> List[Tuple[str, str, str]]:
        """Runs all validation checks and returns a list of results."""
        results = []
        
        # --- UDI & UPC Analysis ---
        # Clean text once for efficiency
        cleaned_text = self.all_text.replace(" ", "").replace("\n", "")
        upcs = set(re.findall(r'(\d{12})', cleaned_text))
        udis = set(re.findall(r'\(01\)(\d{14})', cleaned_text))

        if not upcs and not udis:
            results.append(('failed', "No UPCs or UDIs found across all documents.", "no_serials"))
        else:
            if not upcs:
                results.append(('failed', "UDIs were found, but no matching 12-digit UPCs were detected.", "no_upcs"))
            for upc in upcs:
                # A 14-digit UDI (GTIN-14) often contains a 12-digit UPC padded with leading zeros.
                # We check if the UPC is present in the last 12 digits of any found UDI.
                if not any(upc in udi[2:] for udi in udis):
                    results.append(('failed', f"UPC `{upc}` does not have a matching UDI.", f"mismatch_{upc}"))
                else:
                    results.append(('passed', f"UPC `{upc}` has a matching UDI.", f"match_{upc}"))
        
        # --- Reference Text Check ---
        if self.reference_text:
            if self.reference_text.lower() not in self.all_text.lower():
                results.append(('failed', "Mandatory text from the reference file was NOT found.", "ref_text_missing"))
            else:
                results.append(('passed', "Mandatory text from the reference file was found.", "ref_text_ok"))
            
        return results
