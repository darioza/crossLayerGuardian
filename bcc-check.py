#!/usr/bin/env python3

import sys
import os

def check_bcc():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    try:
        import bcc
        print(f"BCC found and imported successfully. Version: {bcc.__version__}")
        print(f"BCC path: {bcc.__file__}")
    except ImportError as e:
        print(f"Failed to import BCC: {e}")
        print("\nPython path:")
        for path in sys.path:
            print(path)

if __name__ == "__main__":
    check_bcc()
