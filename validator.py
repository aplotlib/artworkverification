import re
import pandas as pd

class ArtworkValidator:
    def __init__(self, checklist_rules, common_errors):
        self.rules = checklist_rules
        self.common_errors = common_errors

    def run_validation(self, extracted_text, filename, ai_analysis_json=None):
        """
        Runs a full validation suite.
        """
        results = {
            "summary": {"pass": 0, "fail": 0, "warning": 0},
            "details": []
        }

        # 1. SKU Consistency Check (Critical Priority based on Error Tracker)
        sku_check = self._check_sku_consistency(filename, extracted_text)
        results['details'].append(sku_check)

        # 2. Country of Origin Check (Standard Compliance)
        origin_check = self._check_phrase(extracted_text, ["Made in China", "Origin: China"], "Country of Origin")
        results['details'].append(origin_check)

        # 3. Checklist Item Validation
        # We check for keywords from the loaded CSV checklist in the extracted text
        for rule in self.rules:
            # Simple keyword matching for now - sophisticated version uses AI
            # If the rule is "Outlined Texts", code can't check that easily, so we mark it for AI or Manual.
            
            # Only auto-check items that are likely text-based
            text_keywords = ["UDI", "UPC", "Website", "HCPCS", "Lot", "Name"]
            if any(k in rule['requirement'] for k in text_keywords):
                has_match = self._fuzzy_search(rule['requirement'], extracted_text)
                status = "PASS" if has_match else "WARNING" # Warning because OCR might fail
                results['details'].append({
                    "category": rule['category'],
                    "check": rule['requirement'],
                    "status": status,
                    "observation": "Found in text" if has_match else "Not found in extracted text (Verify Manually)"
                })

        # 4. AI Insights Integration
        # If AI returned structured JSON, we merge it here
        if ai_analysis_json:
            for ai_finding in ai_analysis_json.get('findings', []):
                results['details'].append({
                    "category": "AI Visual Check",
                    "check": ai_finding.get('check', 'Visual Inspection'),
                    "status": ai_finding.get('status', 'INFO').upper(),
                    "observation": ai_finding.get('observation', '')
                })

        # Calculate Summary
        for item in results['details']:
            if item['status'] == 'PASS': results['summary']['pass'] += 1
            elif item['status'] == 'FAIL': results['summary']['fail'] += 1
            else: results['summary']['warning'] += 1

        return results

    def _check_sku_consistency(self, filename, text):
        """
        Extracts SKU from filename (e.g., 'DMD1001BLK.pdf') and checks if it exists in the text.
        """
        # Regex to find patterns like LVA1001, DMD2020, etc. (3-4 letters followed by numbers)
        filename_sku_match = re.search(r'([A-Z]{3,4}\d{4}[A-Z]*)', filename.upper())
        
        if not filename_sku_match:
            return {
                "category": "Critical",
                "check": "SKU Consistency",
                "status": "WARNING",
                "observation": f"Could not auto-detect SKU from filename: {filename}"
            }
        
        target_sku = filename_sku_match.group(1)
        
        if target_sku in text.upper():
            return {
                "category": "Critical",
                "check": "SKU Consistency",
                "status": "PASS",
                "observation": f"SKU {target_sku} found in both filename and artwork."
            }
        else:
            return {
                "category": "Critical",
                "check": "SKU Consistency",
                "status": "FAIL",
                "observation": f"Filename implies SKU is {target_sku}, but this was NOT found in the artwork text."
            }

    def _check_phrase(self, text, phrases, label):
        found = any(p.upper() in text.upper() for p in phrases)
        return {
            "category": "Compliance",
            "check": label,
            "status": "PASS" if found else "FAIL",
            "observation": f"'{phrases[0]}' found." if found else f"Missing '{phrases[0]}'."
        }

    def _fuzzy_search(self, requirement, text):
        # Very basic implementation - in production, use 'fuzzywuzzy' library
        # Breaking requirement into keywords
        keywords = [w for w in requirement.split() if len(w) > 3]
        if not keywords: return True
        
        hits = 0
        text_upper = text.upper()
        for k in keywords:
            if k.upper() in text_upper:
                hits += 1
        
        # If more than 50% of keywords in the requirement are present, we assume a match
        return (hits / len(keywords)) > 0.5
