def get_direction(route_name, orig_en):
    if not isinstance(route_name, str):
        return 0
    if ' > ' in route_name:
        parts = route_name.split(' > ')
        if len(parts) > 1:
            if parts[0] == orig_en:
                return 0
    return 1