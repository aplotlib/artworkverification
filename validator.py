import re
from typing import List, Tuple, Dict
from collections import defaultdict

class ArtworkValidator:
    """Performs core, rule-based validation checks on the combined text."""

    def __init__(self, all_text: str, reference_text: str = None):
        self.all_text = all_text
        self.reference_text = reference_text

    def validate(self, docs: List[Dict[str, any]]) -> Tuple[List, Dict]:
        """Runs all validation checks and returns global and per-document results."""
        global_results = []
        per_doc_results = defaultdict(list)

        packaging_docs = [d for d in docs if d['doc_type'] == 'packaging_artwork']
        other_docs = [d for d in docs if d['doc_type'] != 'packaging_artwork']

        # --- Primary Validation on Packaging Artwork ---
        if not packaging_docs:
            global_results.append(('failed', "No packaging artwork file was found for primary validation.", "no_packaging"))
        else:
            packaging_text = " ".join(d['text'] for d in packaging_docs).replace("\n", " ")
            packaging_qrs = "".join(qr for d in packaging_docs for qr in d['qr_codes'])
            
            # Use more flexible regex to find UPCs, UDIs, and SKUs
            upcs = set(re.findall(r'\b\d{12}\b', packaging_text))
            udis = set(re.findall(r'\(01\)\d{14}', packaging_text))
            skus = set(re.findall(r'\b(LVA\d{4,}[A-Z]*)\b', packaging_text, re.IGNORECASE))
            
            # Check for the presence of the product name
            product_name_found = "Wheelchair Bag Advanced" in packaging_text

            if not upcs:
                global_results.append(('failed', "No 12-digit UPCs found on packaging artwork.", "no_upc_on_packaging"))
            
            if not udis:
                global_results.append(('failed', "No UDI found on packaging artwork.", "no_udi_on_packaging"))
            
            if not skus:
                global_results.append(('failed', "No SKU found on packaging artwork.", "no_sku_on_packaging"))
            
            if not product_name_found:
                global_results.append(('failed', "Product name 'Wheelchair Bag Advanced' not found on packaging artwork.", "no_product_name_on_packaging"))

            for upc in upcs:
                # Check if the UPC is properly encoded within a UDI
                if not any(upc in udi for udi in udis):
                    global_results.append(('failed', f"Packaging UPC `{upc}` does not have a matching UDI on the packaging.", f"mismatch_{upc}"))
                else:
                    global_results.append(('passed', f"Packaging UPC `{upc}` has a matching UDI.", f"match_{upc}"))
                
                if packaging_qrs and upc not in packaging_qrs:
                    global_results.append(('failed', f"Packaging UPC `{upc}` was not found in any packaging QR codes.", f"qr_mismatch_{upc}"))
                elif packaging_qrs:
                    global_results.append(('passed', f"Packaging UPC `{upc}` was found in a packaging QR code.", f"qr_match_{upc}"))

            # --- Consistency Check against Other Documents ---
            other_text = " ".join(d['text'] for d in other_docs).replace("\n", " ")
            other_upcs = set(re.findall(r'\b\d{12}\b', other_text))
            other_skus = set(re.findall(r'\b(LVA\d{4,}[A-Z]*)\b', other_text, re.IGNORECASE))

            for other_upc in other_upcs:
                if other_upc not in upcs:
                    global_results.append(('failed', f"UPC `{other_upc}` from an auxiliary file does not match any packaging UPC.", "consistency_fail"))
                else:
                     global_results.append(('passed', f"UPC `{other_upc}` in auxiliary file is consistent with packaging.", "consistency_pass"))
            
            for other_sku in other_skus:
                if skus and other_sku not in skus: # Only check if SKUs were found on packaging
                    global_results.append(('failed', f"SKU `{other_sku}` from an auxiliary file does not match any packaging SKU.", "consistency_fail"))
                elif skus:
                    global_results.append(('passed', f"SKU `{other_sku}` in auxiliary file is consistent with packaging.", "consistency_pass"))

        # --- Per-Document Check: Reference Text ---
        if self.reference_text:
            required_phrases = [phrase.strip() for phrase in self.reference_text.split('\n') if phrase.strip()]
            for doc in docs:
                doc_text_lower = doc['text'].lower()
                for phrase in required_phrases:
                    if phrase.lower() not in doc_text_lower:
                        per_doc_results[doc['filename']].append(('failed', f"Missing required text: '{phrase}'"))
                    else:
                        per_doc_results[doc['filename']].append(('passed', f"Found required text: '{phrase}'"))

        return global_results, per_doc_results
