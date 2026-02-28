-- Convert all salaries to annual USD using hardcoded exchange rates
-- Run this after jobs_cleaned table is populated

-- First, add columns if they don't exist
ALTER TABLE jobs_cleaned 
ADD COLUMN IF NOT EXISTS salary_min_annual_usd DECIMAL(14, 2),
ADD COLUMN IF NOT EXISTS salary_max_annual_usd DECIMAL(14, 2),
ADD COLUMN IF NOT EXISTS salary_annual_usd DECIMAL(14, 2);

-- Update salaries using hardcoded exchange rates
UPDATE jobs_cleaned j
SET 
    -- Convert min salary to annual USD
    salary_min_annual_usd = 
        CASE 
            -- Hourly to annual (40hrs/week * 52 weeks)
            WHEN LOWER(j.salary_period) IN ('hour', 'hourly', '/hour', 'per hour', 'hr', '/hr') THEN
                j.salary_min * 2080 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0 -- Assume USD if currency not found
                END
            
            -- Daily to annual (5 days/week * 52 weeks)
            WHEN LOWER(j.salary_period) IN ('day', 'daily', '/day', 'per day') THEN
                j.salary_min * 260 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
            
            -- Weekly to annual
            WHEN LOWER(j.salary_period) IN ('week', 'weekly', '/week', 'per week') THEN
                j.salary_min * 52 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
            
            -- Monthly to annual
            WHEN LOWER(j.salary_period) IN ('month', 'monthly', '/month', 'per month') THEN
                j.salary_min * 12 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
            
            -- Already annual or unknown period - just convert currency
            ELSE
                j.salary_min * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
        END,
    
    -- Convert max salary to annual USD (same logic)
    salary_max_annual_usd = 
        CASE 
            WHEN LOWER(j.salary_period) IN ('hour', 'hourly', '/hour', 'per hour', 'hr', '/hr') THEN
                j.salary_max * 2080 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
            WHEN LOWER(j.salary_period) IN ('day', 'daily', '/day', 'per day') THEN
                j.salary_max * 260 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
            WHEN LOWER(j.salary_period) IN ('week', 'weekly', '/week', 'per week') THEN
                j.salary_max * 52 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
            WHEN LOWER(j.salary_period) IN ('month', 'monthly', '/month', 'per month') THEN
                j.salary_max * 12 * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
            ELSE
                j.salary_max * 
                CASE j.salary_currency
                    WHEN 'USD' THEN 1.0 WHEN 'EUR' THEN 1.10 WHEN 'GBP' THEN 1.27
                    WHEN 'CAD' THEN 0.75 WHEN 'AUD' THEN 0.67 WHEN 'JPY' THEN 0.0075
                    WHEN 'CNY' THEN 0.14 WHEN 'INR' THEN 0.012 WHEN 'PLN' THEN 0.25
                    WHEN 'SEK' THEN 0.095 WHEN 'DKK' THEN 0.15 WHEN 'CHF' THEN 1.12
                    WHEN 'BRL' THEN 0.20 WHEN 'MXN' THEN 0.055 WHEN 'TWD' THEN 0.033
                    WHEN 'KRW' THEN 0.00075 WHEN 'VND' THEN 0.000043 WHEN 'AED' THEN 0.27
                    WHEN 'PHP' THEN 0.018 WHEN 'RUB' THEN 0.011 WHEN 'SGD' THEN 0.74
                    WHEN 'THB' THEN 0.028 WHEN 'IDR' THEN 0.000065 WHEN 'MYR' THEN 0.21
                    WHEN 'ZAR' THEN 0.054 WHEN 'NZD' THEN 0.62 WHEN 'NOK' THEN 0.095
                    WHEN 'CZK' THEN 0.045 WHEN 'HUF' THEN 0.0028 WHEN 'RON' THEN 0.22
                    WHEN 'TRY' THEN 0.031 WHEN 'ILS' THEN 0.27 WHEN 'SAR' THEN 0.27
                    WHEN 'QAR' THEN 0.27 WHEN 'KWD' THEN 3.25 WHEN 'OMR' THEN 2.60
                    WHEN 'BDT' THEN 0.0091 WHEN 'PKR' THEN 0.0036 WHEN 'LKR' THEN 0.0033
                    WHEN 'EGP' THEN 0.021 WHEN 'NGN' THEN 0.00066 WHEN 'KES' THEN 0.0068
                    WHEN 'COP' THEN 0.00024 WHEN 'ARS' THEN 0.0012 WHEN 'CLP' THEN 0.0011
                    WHEN 'PEN' THEN 0.27 WHEN 'VES' THEN 0.0000014
                    ELSE 1.0
                END
        END,
    
    -- Calculate average annual salary (only if both values exist)
    salary_annual_usd = 
        CASE 
            WHEN salary_min_annual_usd IS NOT NULL AND salary_max_annual_usd IS NOT NULL 
            THEN (salary_min_annual_usd + salary_max_annual_usd) / 2
            WHEN salary_min_annual_usd IS NOT NULL THEN salary_min_annual_usd
            WHEN salary_max_annual_usd IS NOT NULL THEN salary_max_annual_usd
            ELSE NULL
        END
WHERE j.salary_min IS NOT NULL OR j.salary_max IS NOT NULL;
