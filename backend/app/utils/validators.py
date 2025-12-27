def validate_external_id(value: str) -> bool:
    if not value or len(value) < 3:
        return False
    return True
