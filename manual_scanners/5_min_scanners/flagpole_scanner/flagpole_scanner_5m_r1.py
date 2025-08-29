#!/usr/bin/env python3
"""
Flagpole Scanner 5M R1 - Database Integration Wrapper
Captures text output from pattern_scanner.py and logs to database
"""

import os
import sys
import subprocess
import importlib.util
from datetime import datetime
from io import StringIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('FlagpoleR1')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Import parser and logger
from database_and_logging.flagpole_parser import FlagpoleOutputParser
from database_and_logging.flagpole_logger import FlagpoleLogger

class FlagpoleScanner5MR1:
    """R1 wrapper that captures R0 output and logs to database"""
    
    def __init__(self):
        self.scanner_name = 'Flagpole 5M Scanner R1'
        self.parser = FlagpoleOutputParser()
        self.logger = FlagpoleLogger(timeframe='5m', min_score=60)
        
        # Path to R0 scanner (with space in folder name)
        self.scanner_path = os.path.join(
            project_root,
            "manual_scanners",
            "5_min_scanners", 
            "flagpole scanner",  # Space preserved
            "pattern_scanner.py"
        )
    
    def capture_scanner_output(self) -> str:
        """
        Execute scanner and capture its text output
        """
        try:
            # Method 1: Try running as subprocess to capture stdout
            result = subprocess.run(
                [sys.executable, self.scanner_path],
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout
                cwd=project_root
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Scanner returned error code {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            logger.error("Scanner execution timed out after 3 minutes")
            return ""
        except Exception as e:
            logger.error(f"Failed to execute scanner: {e}")
            
            # Method 2: Try dynamic import and capture stdout
            return self._try_dynamic_import()
    
    def _try_dynamic_import(self) -> str:
        """
        Fallback: Import module dynamically and capture stdout
        """
        try:
            spec = importlib.util.spec_from_file_location(
                "pattern_scanner", 
                self.scanner_path
            )
            if not spec or not spec.loader:
                logger.error("Failed to create module spec")
                return ""
            
            module = importlib.util.module_from_spec(spec)
            
            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                spec.loader.exec_module(module)
                
                # Try to find and call main function
                if hasattr(module, 'main'):
                    module.main()
                elif hasattr(module, 'run'):
                    module.run()
                elif hasattr(module, 'scan_patterns'):
                    module.scan_patterns()
                else:
                    # Module might execute on import
                    pass
                
                output = captured_output.getvalue()
                
            finally:
                sys.stdout = old_stdout
            
            return output
            
        except Exception as e:
            logger.error(f"Dynamic import failed: {e}")
            return ""
    
    def run(self):
        """Execute scanner, parse output, and log to database"""
        
        print(f"\n{'='*60}")
        print(f"🚩 {self.scanner_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Capture scanner output
        logger.info("Capturing scanner output...")
        output = self.capture_scanner_output()
        
        if not output:
            logger.warning("No output captured from scanner")
            return
        
        # Save raw output for debugging
        debug_file = f"/tmp/flagpole_scanner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(debug_file, 'w') as f:
            f.write(output)
        logger.info(f"Raw output saved to {debug_file}")
        
        # Parse output
        logger.info("Parsing scanner output...")
        patterns = self.parser.parse_scanner_output(output)
        
        if not patterns:
            logger.warning("No patterns found in output")
            return
        
        logger.info(f"Found {len(patterns)} patterns")
        
        # Log to database
        logged = self.logger.log_patterns(patterns)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"🚩 Patterns found: {len(patterns)}")
        print(f"🚩 Logged to DB: {logged}")
        print(f"🚩 Skipped (low score): {len(patterns) - logged}")
        print(f"{'='*60}\n")

def main():
    """Entry point"""
    scanner = FlagpoleScanner5MR1()
    scanner.run()

if __name__ == "__main__":
    main()
