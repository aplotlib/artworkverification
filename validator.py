import re
from typing import List, Tuple, Dict
from collections import defaultdict

class ArtworkValidator:
    """Performs rule-based validation checks based on a primary source."""

    def __init__(self, all_text: str):
        self.all_text = all_text

    def validate(self, docs: List[Dict[str, any]], primary_validation_text: str, primary_validation_file_bytes: bytes = None) -> Tuple[List, Dict]:
        """
        Runs all validation checks against a primary source.

        The primary source is determined in two ways:
        1. If a 'primary_validation_file_bytes' is provided, its text is used.
        2. Otherwise, the user-provided 'primary_validation_text' is used.
        3. If neither is provided, a "best-effort" validation is performed.
        """
        global_results = []
        per_doc_results = defaultdict(list)

        primary_source_text = ""
        primary_source_name = ""

        if primary_validation_file_bytes:
            primary_doc = next((d for d in docs if "Primary_Validation" in d.get('filename', '')), None)
            if primary_doc:
                primary_source_text = primary_doc['text']
                primary_source_name = f"file '{primary_doc['filename']}'"
                docs.remove(primary_doc)
        elif primary_validation_text.strip():
            primary_source_text = primary_validation_text
            primary_source_name = "the user-provided text"
        
        if not primary_source_text:
            global_results.append({'status': 'warning', 'message': "No primary validation source provided. Running in 'best-effort' mode. For best results, please provide a primary validation source.", 'details': "Without a primary source, the system can only check for internal consistency."})
            
            # Best-effort validation
            all_upcs = defaultdict(list)
            all_skus = defaultdict(list)
            for doc in docs:
                doc_text = doc['text'].replace("\n", " ")
                for upc in re.findall(r'\b\d{12}\b', doc_text):
                    all_upcs[upc].append(doc['filename'])
                for sku in re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', doc_text, re.IGNORECASE):
                    all_skus[sku].append(doc['filename'])

            if len(all_upcs) > 1:
                global_results.append({'status': 'failed', 'message': "Multiple UPCs found across documents.", 'details': f"Found UPCs: {', '.join(all_upcs.keys())}"})
            if len(all_skus) > 1:
                global_results.append({'status': 'orange', 'message': "Multiple SKUs found across documents. This may be due to product variations.", 'details': f"Found SKUs: {', '.join(all_skus.keys())}"})
            
            if len(all_upcs) <= 1 and len(all_skus) <= 1:
                global_results.append({'status': 'passed', 'message': "All documents appear to be consistent.", 'details': "Only one SKU and one UPC were found across all documents."})
            
            return global_results, per_doc_results

        # Extract key identifiers from the primary source
        primary_upcs = set(re.findall(r'\b\d{12}\b', primary_source_text))
        primary_udis = set(re.findall(r'\(01\)\d{14}', primary_source_text))
        primary_skus = set(re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', primary_source_text, re.IGNORECASE))
        
        global_results.append({'status': 'info', 'message': f"Validation is being performed against {primary_source_name}.", 'details': f"Primary UPCs: {', '.join(primary_upcs)}, Primary SKUs: {', '.join(primary_skus)}"})


        # --- Consistency Check against Other Documents ---
        for doc in docs:
            doc_text = doc['text'].replace("\n", " ")
            doc_upcs = set(re.findall(r'\b\d{12}\b', doc_text))
            doc_skus = set(re.findall(r'\b(LVA\d{4,}|CSH\d{4,})[A-Z]*\b', doc_text, re.IGNORECASE))

            for upc in doc_upcs:
                if primary_upcs and upc not in primary_upcs:
                    global_results.append({'status': 'failed', 'message': f"UPC `{upc}` from '{doc['filename']}' does not match any UPC in the primary source.", 'details': f"Primary UPCs: {', '.join(primary_upcs)}"})
            
            for sku in doc_skus:
                if primary_skus and sku.upper() not in [s.upper() for s in primary_skus]:
                    global_results.append({'status': 'failed', 'message': f"SKU `{sku}` from '{doc['filename']}' does not match any SKU in the primary source.", 'details': f"Primary SKUs: {', '.join(primary_skus)}"})

        if not any(r['status'] == 'failed' for r in global_results):
             global_results.append({'status': 'passed', 'message': f"All documents are consistent with {primary_source_name}.", 'details': "All SKUs and UPCs match the primary source."})

        return global_results, per_doc_results
