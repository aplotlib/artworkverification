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
            # The text from file_processor is already a string representation of the CSV
            csv_text = primary_doc.get('text', '')
            df = pd.read_csv(StringIO(csv_text), header=None)
            
            # Heuristically find key-value pairs. This is fragile and depends on CSV structure.
            # Example logic: Find the cell containing "PRODUCT SKU CODE" and get the value from a nearby cell.
            # This needs to be adapted based on the actual, consistent structure of your QC sheets.
            
            # A more robust approach might be to find the row containing the key.
            # Let's assume keys are in column 1 and values are in column 4 for this example.
            
            # Simplified extraction logic - you may need to make this more robust
            sku_row = df[df.apply(lambda row: row.astype(str).str.contains('PRODUCT SKU CODE').any(), axis=1)]
            if not sku_row.empty:
                 # This is an example, you might need to find the exact cell location
                self.golden_record['SKU'] = sku_row.iloc[0, 4]

            name_row = df[df.apply(lambda row: row.astype(str).str.contains('PRODUCT NAME').any(), axis=1)]
            if not name_row.empty:
                self.golden_record['ProductName'] = name_row.iloc[0, 2]
                
            color_row = df[df.apply(lambda row: row.astype(str).str.contains('PRODUCT COLOR').any(), axis=1)]
            if not color_row.empty:
                self.golden_record['Color'] = color_row.iloc[0, 4]
                
            if self.golden_record:
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
            if re.search(re.escape(sku_in_record), doc_text, re.IGNORECASE):
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


    def validate(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Orchestrates the validation process.
        1. Finds the primary QC sheet.
        2. Creates the golden record.
        3. Validates all other documents against it.
        """
        # Find the primary QC sheet first
        primary_doc = next((doc for doc in docs if doc.get('doc_type') == 'qc_sheet'), None)

        if not primary_doc:
            self.validation_results.append({
                'status': 'failed', 'check_name': 'Setup',
                'message': "❌ No QC Sheet found to use as the source of truth.",
                'details': "Please upload a file that can be identified as a 'qc_sheet' based on config.py."
            })
            return self.validation_results

        self.create_golden_record(primary_doc)

        # If golden record creation failed, stop
        if not self.golden_record:
            return self.validation_results
            
        # Validate all other documents
        for doc in docs:
            if doc != primary_doc: # Don't validate the source against itself
                self.validate_document(doc)

        return self.validation_results
