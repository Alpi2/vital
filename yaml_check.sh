#!/bin/bash

# Simple YAML syntax checker
echo "🔍 Checking YAML syntax..."

if command -v python3 &> /dev/null; then
    python3 -c "
import sys
import json

def simple_yaml_check(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        # Basic YAML structure checks
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for tabs (YAML doesn't allow tabs)
            if '\t' in line:
                print(f'❌ Line {i}: Tab character found')
                return False
            
            # Check for common syntax issues
            if line.strip() and not line.startswith('#'):
                if ':' in line and not line.strip().endswith(':'):
                    parts = line.split(':', 1)
                    if len(parts) > 1 and parts[1].strip() and not parts[1].startswith(' '):
                        print(f'❌ Line {i}: Missing space after colon')
                        return False
        
        print('✅ YAML syntax appears valid')
        return True
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

simple_yaml_check('$1')
    " "$1"
else
    echo "❌ Python3 not available for YAML checking"
    exit 1
fi
