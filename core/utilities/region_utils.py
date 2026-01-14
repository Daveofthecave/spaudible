# core/utilities/region_utils.py

# Region dictionary: 8 geographical/cultural/linguistic regions
REGION_MAPPING = {
    0: ["AU", "CA", "CB", "GB", "GG", "GX", "IE", "IM", "JE", "NZ", 
        "QM", "QT", "QZ", "UK", "US"],  # Anglo
    1: ["AD", "AT", "BE", "CH", "DE", "DK", "EE", "FI", "FO", "FR", 
        "FX", "GI", "GL", "IS", "IT", "LI", "LU", "MC", "MT", "NL", 
        "NO", "PT", "SE", "SM"],  # Western European
    2: ["AL", "BA", "BG", "BY", "CS", "CY", "CZ", "GR", "HR", "HU", 
        "LT", "LV", "MD", "ME", "MK", "PL", "RO", "RS", "RU", "SI", 
        "SK", "UA", "XK", "YU"],  # Eastern European
    3: ["AR", "BC", "BK", "BO", "BP", "BR", "BX", "BZ", "CL", "CO", 
        "CR", "CU", "DO", "EC", "ES", "GT", "HN", "MX", "NI", "PA", 
        "PE", "PR", "PY", "SV", "UY", "VE"],  # Hispanic
    4: ["BN", "CN", "HK", "ID", "JP", "KG", "KH", "KR", "KS", "KZ", 
        "LA", "MM", "MN", "MO", "MY", "PG", "PH", "SG", "TH", "TL", 
        "TW", "UZ", "VN"],  # Asian
    5: ["BD", "BT", "IN", "LK", "MV", "NP", "PK"],  # Indian
    6: ["AE", "AF", "AM", "AZ", "BH", "DZ", "EG", "GE", "IL", "IQ", 
        "IR", "JO", "KW", "LB", "MA", "OM", "PS", "QA", "SA", "SY", 
        "TN", "TR", "YE"],  # Middle Eastern
    7: ["AG", "AI", "AO", "AW", "BB", "BF", "BI", "BJ", "BM", "BS", 
        "BW", "CD", "CF", "CG", "CI", "CM", "CP", "CV", "CW", "DG", 
        "DM", "ET", "FJ", "GA", "GD", "GH", "GM", "GN", "GQ", "GY", 
        "HT", "JM", "KE", "KM", "KN", "KY", "LC", "LR", "LS", "MF", 
        "MG", "ML", "MP", "MR", "MS", "MU", "MW", "MZ", "NA", "NE", 
        "NG", "PF", "QN", "RW", "SB", "SC", "SD", "SL", "SN", "SO", 
        "SR", "SS", "ST", "SX", "SZ", "TC", "TD", "TG", "TO", "TT", 
        "TZ", "UG", "VC", "VG", "VU", "VV", "ZA", "ZB", "ZM", "ZW", 
        "ZZ"]  # Other
}

def get_region_from_isrc(isrc: str) -> int:
    """
    Convert ISRC to region index (0-7).
    
    Args:
        isrc: ISRC code (first 2 characters are country code)
        
    Returns:
        Region index (0-7), defaults to 7 (Other)
    """
    if not isrc or not isrc.strip():
        return 7
    if len(isrc) < 2:
        return 7
    country_code = isrc[:2].upper()
    for region_id, countries in REGION_MAPPING.items():
        if country_code in countries:
            return region_id
    return 7