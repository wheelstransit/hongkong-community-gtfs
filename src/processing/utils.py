def get_direction(route_name, orig_en):
    if not isinstance(route_name, str):
        return 0
    if ' > ' in route_name:
        parts = route_name.split(' > ')
        if len(parts) > 1:
            if parts[0] == orig_en:
                return 0
    return 1

def smart_title_case(text):
    if not isinstance(text, str) or not text:
        return text
    
    upper_count = sum(1 for c in text if c.isupper())
    if upper_count < len(text) * 0.5:
        return text
    
    preserve_upper = {
        'MTR', 'KMB', 'CTB', 'GMB', 'NLB', 'NWFB',
        'IFC', 'ICC', 'HSBC', 'AIA', 'ATM', 'WC',
        'I', 'II', 'III', 'IV', 'V',
    }
    
    keep_lower = {'of', 'the', 'and', 'at', 'in', 'on', 'to', 'for', 'with', 'by'}
    
    def process_word(word, is_first):
        if word.upper() in preserve_upper:
            return word.upper()
        if not is_first and word.lower() in keep_lower:
            return word.lower()
        
        return word.capitalize()
    
    import re
    
    parts = []
    current = []
    in_parens = False
    
    for char in text:
        if char == '(':
            if current:
                parts.append(('text', ''.join(current)))
                current = []
            in_parens = True
            current.append(char)
        elif char == ')':
            current.append(char)
            parts.append(('parens', ''.join(current)))
            current = []
            in_parens = False
        else:
            current.append(char)
    
    if current:
        parts.append(('text', ''.join(current)))
    
    result_parts = []
    is_first_word = True
    
    for part_type, part_text in parts:
        if part_type == 'parens':
            inner = part_text[1:-1]
            words = inner.split()
            processed_inner = ' '.join(process_word(w, i == 0) for i, w in enumerate(words))
            result_parts.append(f'({processed_inner})')
        else:
            words = part_text.split()
            for word in words:
                if '-' in word:
                    hyphen_parts = word.split('-')
                    processed_hyphen = '-'.join(process_word(hp, is_first_word and i == 0) 
                                                for i, hp in enumerate(hyphen_parts))
                    result_parts.append(processed_hyphen)
                else:
                    result_parts.append(process_word(word, is_first_word))
                is_first_word = False
    
    return ' '.join(result_parts)