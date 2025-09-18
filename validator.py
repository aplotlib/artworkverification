import re
from typing import List, Tuple, Dict
from collections import defaultdict

class ArtworkValidator:
    """Performs core, rule-based validation checks on the combined text."""

    def __init__(self, all_text: str):
        self.all_text = all_text

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

            upcs = set(re.findall(r'\b\d{12}\b', packaging_text))
            udis = set(re.findall(r'\(01\)\d{14}', packaging_text))
            skus = set(re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', packaging_text, re.IGNORECASE))

            if not upcs:
                global_results.append(('failed', "No 12-digit UPCs found on packaging artwork.", "no_upc_on_packaging"))
            if not udis:
                global_results.append(('failed', "No UDI found on packaging artwork.", "no_udi_on_packaging"))
            if not skus:
                global_results.append(('failed', "No SKU found on packaging artwork.", "no_sku_on_packaging"))

            for upc in upcs:
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
            other_skus = set(re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', other_text, re.IGNORECASE))

            for other_upc in other_upcs:
                if upcs and other_upc not in upcs:
                    global_results.append(('failed', f"UPC `{other_upc}` from an auxiliary file does not match any packaging UPC.", "consistency_fail"))
            for other_sku in other_skus:
                if skus and other_sku not in skus:
                    global_results.append(('failed', f"SKU `{other_sku}` from an auxiliary file does not match any packaging SKU.", "consistency_fail"))
        
        return global_results, per_doc_results
