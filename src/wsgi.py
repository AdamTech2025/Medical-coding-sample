"""
WSGI config for the project.

This file imports and re-exports the WSGI application from the medical module.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the application from medical/wsgi.py
from medical.wsgi import application

# This is what Vercel will use
app = application 