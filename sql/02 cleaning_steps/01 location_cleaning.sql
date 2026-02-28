-- Country Cleaning Script for jobs_cleaned table
-- Run this after creating jobs_cleaned table

-- First, create a backup of original country values (optional but recommended)
ALTER TABLE jobs_cleaned ADD COLUMN IF NOT EXISTS country_original VARCHAR(255);
UPDATE jobs_cleaned SET country_original = country;

-- United States variations
UPDATE jobs_cleaned SET country = 'United States' WHERE UPPER(country) IN ('US', 'USA', 'U.S.', 'UNITED STATES', 'UNITED STATES OF AMERICA', 'CA US');

-- US States (all 50 states + DC)
UPDATE jobs_cleaned SET country = 'United States' WHERE UPPER(country) IN (
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 
    'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 
    'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
);

-- United Kingdom variations
UPDATE jobs_cleaned SET country = 'United Kingdom' WHERE UPPER(country) IN ('UK', 'UNITED KINGDOM', 'ENGLAND');

-- France variations
UPDATE jobs_cleaned SET country = 'France' WHERE UPPER(country) IN ('FR', 'FRANCE');

-- South Africa variations
UPDATE jobs_cleaned SET country = 'South Africa' WHERE UPPER(country) IN ('ZA', 'SOUTH AFRICA');

-- Cyprus variations
UPDATE jobs_cleaned SET country = 'Cyprus' WHERE UPPER(country) IN ('CY', 'CYPRUS');

-- Ukraine variations
UPDATE jobs_cleaned SET country = 'Ukraine' WHERE UPPER(country) IN ('UA', 'UKRAINE');

-- Italy variations
UPDATE jobs_cleaned SET country = 'Italy' WHERE UPPER(country) IN ('IT', 'ITALY');

-- Morocco variations
UPDATE jobs_cleaned SET country = 'Morocco' WHERE UPPER(country) IN ('MAROC', 'MOROCCO');

-- Czech Republic variations
UPDATE jobs_cleaned SET country = 'Czech Republic' WHERE UPPER(country) IN ('CZECHIA', 'CZECH REPUBLIC');

-- Bosnia and Herzegovina variations
UPDATE jobs_cleaned SET country = 'Bosnia and Herzegovina' WHERE UPPER(country) IN ('BOSNIA', 'BOSNIA AND HERZEGOVINA');

-- South Korea variations
UPDATE jobs_cleaned SET country = 'South Korea' WHERE UPPER(country) IN ('KOREA', 'SOUTH KOREA', 'KOREA, REPUBLIC OF');

-- Turkey variations
UPDATE jobs_cleaned SET country = 'Turkey' WHERE UPPER(country) IN ('TURKIYE', 'TURKEY', 'TR');

-- Tunisia variations
UPDATE jobs_cleaned SET country = 'Tunisia' WHERE UPPER(country) IN ('TUNISIE', 'TUNISIA');

-- Brazil variations
UPDATE jobs_cleaned SET country = 'Brazil' WHERE UPPER(country) IN ('BRASIL', 'BRAZIL', 'BR');

-- Japan variations
UPDATE jobs_cleaned SET country = 'Japan' WHERE UPPER(country) IN ('JP', 'JAPAN');

-- Uruguay variations
UPDATE jobs_cleaned SET country = 'Uruguay' WHERE UPPER(country) IN ('UY', 'URUGUAY');

-- Philippines variations
UPDATE jobs_cleaned SET country = 'Philippines' WHERE UPPER(country) IN ('PH', 'PHILIPPINES');

-- Singapore variations
UPDATE jobs_cleaned SET country = 'Singapore' WHERE UPPER(country) IN ('SG', 'SINGAPORE');

-- Netherlands variations
UPDATE jobs_cleaned SET country = 'Netherlands' WHERE UPPER(country) IN ('NL', 'NETHERLANDS', 'THE NETHERLANDS');

-- Sweden variations
UPDATE jobs_cleaned SET country = 'Sweden' WHERE UPPER(country) IN ('SE', 'SWEDEN');

-- Switzerland variations
UPDATE jobs_cleaned SET country = 'Switzerland' WHERE UPPER(country) IN ('CH', 'SWITZERLAND');

-- Greece variations
UPDATE jobs_cleaned SET country = 'Greece' WHERE UPPER(country) IN ('GR', 'GREECE');

-- India variations
UPDATE jobs_cleaned SET country = 'India' WHERE UPPER(country) IN ('IN', 'INDIA');

-- Portugal variations
UPDATE jobs_cleaned SET country = 'Portugal' WHERE UPPER(country) IN ('PT', 'PORTUGAL');

-- Australia variations
UPDATE jobs_cleaned SET country = 'Australia' WHERE UPPER(country) IN ('AU', 'AUSTRALIA');

-- New Zealand variations
UPDATE jobs_cleaned SET country = 'New Zealand' WHERE UPPER(country) IN ('NZ', 'NEW ZEALAND');

-- Belgium variations
UPDATE jobs_cleaned SET country = 'Belgium' WHERE UPPER(country) IN ('BE', 'BELGIUM');

-- Mexico variations
UPDATE jobs_cleaned SET country = 'Mexico' WHERE UPPER(country) IN ('MX', 'MEXICO');

-- Russia variations
UPDATE jobs_cleaned SET country = 'Russia' WHERE UPPER(country) IN ('RU', 'RUSSIA');

-- Egypt variations
UPDATE jobs_cleaned SET country = 'Egypt' WHERE UPPER(country) IN ('EG', 'EGYPT');

-- Finland variations
UPDATE jobs_cleaned SET country = 'Finland' WHERE UPPER(country) IN ('FI', 'FINLAND');

-- Saudi Arabia variations
UPDATE jobs_cleaned SET country = 'Saudi Arabia' WHERE UPPER(country) IN ('SA', 'SAUDI ARABIA');

-- Ireland variations
UPDATE jobs_cleaned SET country = 'Ireland' WHERE UPPER(country) IN ('IE', 'IRELAND');

-- Germany variations
UPDATE jobs_cleaned SET country = 'Germany' WHERE UPPER(country) IN ('DE', 'GERMANY');

-- Spain variations
UPDATE jobs_cleaned SET country = 'Spain' WHERE UPPER(country) IN ('ES', 'SPAIN');

-- China variations
UPDATE jobs_cleaned SET country = 'China' WHERE UPPER(country) IN ('CN', 'CHINA');

-- Canada variations
UPDATE jobs_cleaned SET country = 'Canada' WHERE UPPER(country) IN ('CA', 'CANADA');

-- United Arab Emirates
UPDATE jobs_cleaned SET country = 'United Arab Emirates' WHERE UPPER(country) = 'UAE';

-- Handle continent/region entries
UPDATE jobs_cleaned SET country = 'Europe' WHERE UPPER(country) = 'EUROPE';
UPDATE jobs_cleaned SET country = 'North America' WHERE UPPER(country) = 'NORTH AMERICA';
UPDATE jobs_cleaned SET country = 'Latin America' WHERE UPPER(country) = 'LATIN AMERICA';

-- Handle unknown/empty values
UPDATE jobs_cleaned SET country = 'Unknown' WHERE country IS NULL OR TRIM(country) = '' OR UPPER(country) IN ('[ ]', 'UNKNOWN');
 
-- Convert all countries to title case for consistency
UPDATE jobs_cleaned SET country = INITCAP(country);

-- Remove any leading/trailing whitespace that could cause duplicates
UPDATE jobs_cleaned SET country = TRIM(country); 

-- Show summary of changes
SELECT 
    country_original,
    country,
    COUNT(*) as count
FROM jobs_cleaned 
WHERE country_original != country OR country_original IS NULL
GROUP BY country_original, country
ORDER BY count DESC
LIMIT 20;