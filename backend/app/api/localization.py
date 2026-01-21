"""Localization API endpoints for i18n support."""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any
import json
import os
from pathlib import Path

router = APIRouter(prefix="/api/i18n", tags=["localization"])


@router.get("/languages")
async def get_available_languages() -> List[Dict[str, str]]:
    """Get list of available languages."""
    return [
        {"code": "tr", "name": "Türkçe", "flag": "🇹🇷", "locale": "tr_TR"},
        {"code": "en", "name": "English", "flag": "🇬🇧", "locale": "en_US"},
        {"code": "de", "name": "Deutsch", "flag": "🇩🇪", "locale": "de_DE"},
        {"code": "fr", "name": "Français", "flag": "🇫🇷", "locale": "fr_FR"},
        {"code": "es", "name": "Español", "flag": "🇪🇸", "locale": "es_ES"},
        {"code": "ar", "name": "العربية", "flag": "🇸🇦", "locale": "ar_SA"},
        {"code": "zh", "name": "中文", "flag": "🇨🇳", "locale": "zh_CN"},
        {"code": "ru", "name": "Русский", "flag": "🇷🇺", "locale": "ru_RU"},
        {"code": "ja", "name": "日本語", "flag": "🇯🇵", "locale": "ja_JP"},
        {"code": "pt", "name": "Português", "flag": "🇵🇹", "locale": "pt_PT"}
    ]


@router.get("/translations/{lang_code}")
async def get_translations(lang_code: str) -> Dict[str, Any]:
    """Get translations for a specific language."""
    
    # Language code to locale mapping
    lang_map = {
        "tr": "tr_TR",
        "en": "en_US",
        "de": "de_DE",
        "fr": "fr_FR",
        "es": "es_ES",
        "ar": "ar_SA",
        "zh": "zh_CN",
        "ru": "ru_RU",
        "ja": "ja_JP",
        "pt": "pt_PT"
    }
    
    locale = lang_map.get(lang_code, "en_US")
    
    # Get the project root directory
    current_file = Path(__file__)
    backend_dir = current_file.parent.parent.parent  # Go up to backend/
    project_root = backend_dir.parent  # Go up to project root
    localization_path = project_root / "localization" / locale
    
    translations = {}
    
    # Check if localization directory exists
    if not localization_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Translations not found for language: {lang_code}"
        )
    
    # Load all JSON files from the locale directory
    try:
        for json_file in localization_path.glob("*.json"):
            module_name = json_file.stem  # filename without extension
            with open(json_file, 'r', encoding='utf-8') as f:
                translations[module_name] = json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading translations: {str(e)}"
        )
    
    return translations


@router.get("/translations/{lang_code}/{module}")
async def get_module_translations(lang_code: str, module: str) -> Dict[str, Any]:
    """Get translations for a specific module in a language."""
    
    lang_map = {
        "tr": "tr_TR",
        "en": "en_US",
        "de": "de_DE",
        "fr": "fr_FR",
        "es": "es_ES",
        "ar": "ar_SA",
        "zh": "zh_CN",
        "ru": "ru_RU",
        "ja": "ja_JP",
        "pt": "pt_PT"
    }
    
    locale = lang_map.get(lang_code, "en_US")
    
    # Get the project root directory
    current_file = Path(__file__)
    backend_dir = current_file.parent.parent.parent
    project_root = backend_dir.parent
    translation_file = project_root / "localization" / locale / f"{module}.json"
    
    if not translation_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Module '{module}' not found for language: {lang_code}"
        )
    
    try:
        with open(translation_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading module translations: {str(e)}"
        )
