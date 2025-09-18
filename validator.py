import re
from typing import List, Tuple, Dict
from collections import defaultdict

class ArtworkValidator:
    """Performs rule-based validation checks based on a primary source."""

    def __init__(self, all_text: str):
        self.all_text = all_text

    def validate(self, docs: List[Dict[str, any]], primary_validation_text: str) -> Tuple[List, Dict]:
        """
        Runs all validation checks against a primary source.

        The primary source is determined in two ways:
        1. If a file named 'Primary_Validation_*.pdf' is found, its text is used.
        2. Otherwise, the user-provided 'primary_validation_text' is used.
        """
        global_results = []
        per_doc_results = defaultdict(list)

        primary_source_text = ""
        primary_source_name = ""

        # 1. Check for a Primary_Validation file
        primary_doc = next((d for d in docs if "Primary_Validation" in d.get('filename', '')), None)

        if primary_doc:
            primary_source_text = primary_doc['text']
            primary_source_name = f"file '{primary_doc['filename']}'"
            docs.remove(primary_doc) # Remove it from the list of docs to be checked
        elif primary_validation_text.strip():
            primary_source_text = primary_validation_text
            primary_source_name = "the user-provided text"
        
        if not primary_source_text:
            global_results.append(('failed', "No primary validation source was provided or found. Skipping rule-based checks.", "no_primary_source"))
            return global_results, per_doc_results

        # Extract key identifiers from the primary source
        primary_upcs = set(re.findall(r'\b\d{12}\b', primary_source_text))
        primary_udis = set(re.findall(r'\(01\)\d{14}', primary_source_text))
        primary_skus = set(re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', primary_source_text, re.IGNORECASE))
        
        global_results.append(('passed', f"Validation is being performed against {primary_source_name}.", "primary_source_info"))


        # --- Consistency Check against Other Documents ---
        for doc in docs:
            doc_text = doc['text'].replace("\n", " ")
            doc_upcs = set(re.findall(r'\b\d{12}\b', doc_text))
            doc_skus = set(re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', doc_text, re.IGNORECASE))

            for upc in doc_upcs:
                if primary_upcs and upc not in primary_upcs:
                    global_results.append(('failed', f"UPC `{upc}` from '{doc['filename']}' does not match any UPC in the primary source.", "consistency_fail_upc"))
            
            for sku in doc_skus:
                if primary_skus and sku.upper() not in [s.upper() for s in primary_skus]:
                    global_results.append(('failed', f"SKU `{sku}` from '{doc['filename']}' does not match any SKU in the primary source.", "consistency_fail_sku"))

        if not global_results:
             global_results.append(('passed', f"All documents are consistent with {primary_source_name}.", "consistency_pass"))

        return global_results, per_doc_results
