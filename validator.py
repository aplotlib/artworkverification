import re
import pandas as pd
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from io import StringIO

class ArtworkValidator:
    """Performs rule-based validation by creating a 'golden record' from a primary QC sheet."""

    def __init__(self):
        self.golden_record: Dict[str, Any] = {}
        self.validation_results: List[Dict[str, Any]] = []

    def create_golden_record(self, primary_doc: Dict[str, Any]):
        """
        Populates the golden_record from a primary QC document (CSV).
        """
        try:
            csv_text = primary_doc.get('text', '')
            df = pd.read_csv(StringIO(csv_text), header=None)
            
            # Find the first row which contains all the header info
            first_row = df.iloc[1]

            # Find the cell indices for keys and then get their values
            # This is more robust than fixed indices.
            def get_value_by_key(key):
                try:
                    # Find the index of the cell containing the key
                    key_index = first_row[first_row.astype(str).str.contains(key, na=False)].index[0]
                    # The value is typically a few cells to the right.
                    # Based on your CSV, it's 4 cells over for SKU and Color, and 2 for Name.
                    offset = 4 if key != 'PRODUCT NAME' else 2
                    return first_row.iloc[key_index + offset]
                except (IndexError, KeyError):
                    return None

            self.golden_record['SKU'] = get_value_by_key('PRODUCT SKU CODE')
            self.golden_record['ProductName'] = get_value_by_key('PRODUCT NAME')
            self.golden_record['Color'] = get_value_by_key('PRODUCT COLOR')

            if self.golden_record and self.golden_record['SKU']:
                self.validation_results.append({
                    'status': 'info', 'check_name': 'Golden Record Creation',
                    'message': f"Successfully created Golden Record from '{primary_doc['filename']}'.",
                    'details': f"Record: {self.golden_record}"
                })
            else:
                 self.validation_results.append({
                    'status': 'failed', 'check_name': 'Golden Record Creation',
                    'message': f"Could not create Golden Record from '{primary_doc['filename']}'.",
                    'details': "Check file structure and content."
                })

        except Exception as e:
            self.validation_results.append({
                'status': 'failed', 'check_name': 'Golden Record Creation',
                'message': f"Failed to parse QC sheet '{primary_doc['filename']}'.",
                'details': str(e)
            })

    def validate_document(self, doc: Dict[str, Any]):
        """
        Runs a series of checks on a single document against the golden record.
        """
        doc_text = doc.get('text', '')
        filename = doc.get('filename', 'Unknown File')

        # Check 1: SKU Consistency
        sku_in_record = self.golden_record.get('SKU')
        if sku_in_record:
            if re.search(re.escape(str(sku_in_record)), doc_text, re.IGNORECASE):
                self.validation_results.append({
                    'status': 'passed', 'check_name': 'SKU Consistency',
                    'message': f"✅ SKU '{sku_in_record}' found in '{filename}'.",
                    'confidence': 'high'
                })
            else:
                self.validation_results.append({
                    'status': 'failed', 'check_name': 'SKU Consistency',
                    'message': f"❌ SKU '{sku_in_record}' not found in '{filename}'.",
                    'confidence': 'high'
                })

        # Check 2: Product Name Presence
        product_name_in_record = self.golden_record.get('ProductName')
        if product_name_in_record:
            # Use a simplified version for matching (e.g., ignore 'Advanced')
            short_name = product_name_in_record.replace('Advanced', '').strip()
            if re.search(re.escape(short_name), doc_text, re.IGNORECASE):
                self.validation_results.append({
                    'status': 'passed', 'check_name': 'Product Name',
                    'message': f"✅ Product Name '{short_name}' found in '{filename}'.",
                    'confidence': 'high'
                })
            else:
                self.validation_results.append({
                    'status': 'warning', 'check_name': 'Product Name',
                    'message': f"⚠️ Product Name '{short_name}' not found in '{filename}'.",
                    'confidence': 'medium'
                })

    def validate(self, docs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict]:
        """
        Orchestrates the validation process.
        1. Finds the primary QC sheet.
        2. Creates the golden record.
        3. Validates all other documents against it.
        """
        primary_doc = next((doc for doc in docs if doc.get('doc_type') == 'qc_sheet'), None)

        if not primary_doc:
            self.validation_results.append({
                'status': 'failed', 'check_name': 'Setup',
                'message': "❌ No QC Sheet found to use as the source of truth.",
                'details': "Please upload a file that can be identified as a 'qc_sheet' based on config.py."
            })
            return self.validation_results, {}

        self.create_golden_record(primary_doc)

        if not self.golden_record or not self.golden_record.get('SKU'):
            return self.validation_results, {}
            
        for doc in docs:
            if doc != primary_doc:
                self.validate_document(doc)

        return self.validation_results, {} # Returning empty dict for per_doc_results
