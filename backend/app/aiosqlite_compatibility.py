
"""
aiosqlite compatibility layer for Python 3.14+
Provides sqlite3-like interface using sqlite3
"""

import sqlite3
import warnings
from typing import Any, List, Optional, Union

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning")

class Connection(sqlite3.Connection):
    """Enhanced connection class with aiosqlite compatibility"""
    
    def __init__(self, *args, **kwargs):
        # Convert aiosqlite URLs to sqlite3
        if args and isinstance(args[0], str) and 'aiosqlite://' in args[0]:
            new_args = list(args)
            new_args[0] = args[0].replace('aiosqlite://', 'sqlite://')
            super().__init__(*new_args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
    
    def execute(self, query: str, params: Optional[Any] = None) -> sqlite3.Cursor:
        """Execute query with aiosqlite compatibility"""
        # Handle aiosqlite-specific syntax if needed
        return super().execute(query, params)
    
    def executemany(self, query: str, params: List[Any]) -> sqlite3.Cursor:
        """Execute many queries with aiosqlite compatibility"""
        return super().executemany(query, params)

def connect(database: str, *args, **kwargs):
    """Connect function with aiosqlite compatibility"""
    # Convert aiosqlite URLs to sqlite3
    if 'aiosqlite://' in database:
        database = database.replace('aiosqlite://', 'sqlite://')
    
    return sqlite3.connect(database, *args, **kwargs)

# Create module-level compatibility
def patch_sqlite3():
    """Patch sqlite3 module to support aiosqlite URLs"""
    
    # Save original connect function
    original_connect = sqlite3.connect
    
    def patched_connect(database, *args, **kwargs):
        if isinstance(database, str) and 'aiosqlite://' in database:
            database = database.replace('aiosqlite://', 'sqlite://')
        
        return original_connect(database, *args, **kwargs)
    
    sqlite3.connect = patched_connect
    sqlite3.Connection = Connection

# Apply the patch
patch_sqlite3()

# Export compatibility functions
__all__ = ['connect', 'Connection', 'patch_sqlite3']
