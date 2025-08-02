def get_direction(route_name):
    if not isinstance(route_name, str):
        return 0
    if ' > ' in route_name:
        parts = route_name.split(' > ')
        if len(parts) > 1:
            return 0
    return 1