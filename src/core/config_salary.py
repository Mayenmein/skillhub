# Comprehensive currency mapping
COUNTRY_CURRENCY = {
    'United States': 'USD', 'Canada': 'CAD', 'United Kingdom': 'GBP',
    'Germany': 'EUR', 'France': 'EUR', 'Italy': 'EUR', 'Spain': 'EUR',
    'Netherlands': 'EUR', 'Poland': 'PLN', 'India': 'INR', 'Japan': 'JPY',
    'Australia': 'AUD', 'Brazil': 'BRL', 'Mexico': 'MXN', 'Sweden': 'SEK',
    'Denmark': 'DKK', 'Switzerland': 'CHF', 'China': 'CNY', 'Taiwan': 'TWD',
    'Korea': 'KRW', 'Vietnam': 'VND', 'United Arab Emirates': 'AED',
    'USA': 'USD', 'UK': 'GBP', 'UAE': 'AED', 'US': 'USD',
    'Philippines': 'PHP', 'Russia': 'RUB', 'Singapore': 'SGD',
    'Thailand': 'THB', 'Indonesia': 'IDR', 'Malaysia': 'MYR',
    'South Africa': 'ZAR', 'New Zealand': 'NZD', 'Norway': 'NOK',
    'Finland': 'EUR', 'Belgium': 'EUR', 'Austria': 'EUR',
    'Portugal': 'EUR', 'Ireland': 'EUR', 'Czech Republic': 'CZK',
    'Hungary': 'HUF', 'Romania': 'RON', 'Greece': 'EUR',
    'Turkey': 'TRY', 'Israel': 'ILS', 'Saudi Arabia': 'SAR',
    'Qatar': 'QAR', 'Kuwait': 'KWD', 'Oman': 'OMR',
    'Bangladesh': 'BDT', 'Pakistan': 'PKR', 'Sri Lanka': 'LKR',
    'Egypt': 'EGP', 'Nigeria': 'NGN', 'Kenya': 'KES',
    'Colombia': 'COP', 'Argentina': 'ARS', 'Chile': 'CLP',
    'Peru': 'PEN', 'Venezuela': 'VES'
}

# Updated exchange rates (1 unit of foreign currency = X USD)
EXCHANGE_RATES = {
    'USD': 1.0, 'EUR': 1.10, 'GBP': 1.27, 'CAD': 0.75, 'AUD': 0.67,
    'JPY': 0.0075, 'CNY': 0.14, 'INR': 0.012, 'PLN': 0.25, 'SEK': 0.095,
    'DKK': 0.15, 'CHF': 1.12, 'BRL': 0.20, 'MXN': 0.055, 'TWD': 0.033,
    'KRW': 0.00075, 'VND': 0.000043, 'AED': 0.27, 'PHP': 0.018,
    'RUB': 0.011, 'SGD': 0.74, 'THB': 0.028, 'IDR': 0.000065,
    'MYR': 0.21, 'ZAR': 0.054, 'NZD': 0.62, 'NOK': 0.095,
    'CZK': 0.045, 'HUF': 0.0028, 'RON': 0.22, 'TRY': 0.031,
    'ILS': 0.27, 'SAR': 0.27, 'QAR': 0.27, 'KWD': 3.25,
    'OMR': 2.60, 'BDT': 0.0091, 'PKR': 0.0036, 'LKR': 0.0033,
    'EGP': 0.021, 'NGN': 0.00066, 'KES': 0.0068, 'COP': 0.00024,
    'ARS': 0.0012, 'CLP': 0.0011, 'PEN': 0.27, 'VES': 0.0000014
}

skip_patterns = [
        'market rate', 'competitive', 'negotiable', 'as per company',
        'not specified', 'upon agreement', 'to be discussed', 'n/a',
        'na', 'not disclosed', 'confidential', 'upon experience',
        'depending on experience', 'doe'
    ]


currency_patterns = {
        'usd': 'USD', 'dollar': 'USD', '$': 'USD',
        'eur': 'EUR', 'euro': 'EUR', '€': 'EUR',
        'gbp': 'GBP', 'pound': 'GBP', '£': 'GBP',
        'jpy': 'JPY', 'yen': 'JPY', '¥': 'JPY',
        'inr': 'INR', 'rupee': 'INR', '₹': 'INR',
        'cad': 'CAD', 'canadian': 'CAD', 'c$': 'CAD',
        'aud': 'AUD', 'australian': 'AUD', 'a$': 'AUD',
        'mxn': 'MXN', 'peso': 'MXN', 'mexican peso': 'MXN',
        'php': 'PHP', 'philippine peso': 'PHP', '₱': 'PHP',
        'cny': 'CNY', 'yuan': 'CNY', 'rmb': 'CNY',
        'pln': 'PLN', 'złoty': 'PLN', 'zloty': 'PLN',
        'sek': 'SEK', 'swedish krona': 'SEK', 'kr': 'SEK',
        'dkk': 'DKK', 'danish krone': 'DKK',
        'chf': 'CHF', 'swiss franc': 'CHF',
        'brl': 'BRL', 'real': 'BRL', 'r$': 'BRL',
        'twd': 'TWD', 'nt$': 'TWD', 'taiwan dollar': 'TWD',
        'krw': 'KRW', 'won': 'KRW', '₩': 'KRW',
        'vnd': 'VND', 'dong': 'VND', '₫': 'VND',
        'aed': 'AED', 'dirham': 'AED', 'د.إ': 'AED'
    }