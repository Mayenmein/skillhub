-- Country Cleaning Script for jobs_cleaned table
-- Run this after creating jobs_cleaned table

ALTER TABLE jobs_cleaned ADD COLUMN IF NOT EXISTS country_original VARCHAR(255);
UPDATE jobs_cleaned SET country_original = country;

-- United States variations
UPDATE jobs_cleaned
SET country = 'United States'
WHERE LOWER(country) ~
      '(^|[^a-z])(usa|u[[:space:].]*s[[:space:].]*a|u[[:space:].]*s|united[[:space:]]+states|united[[:space:]]+states[[:space:]]+of[[:space:]]+america)([^a-z]|$)';

-- US States (all 50 states + DC)
UPDATE jobs_cleaned
SET country = 'United States'
WHERE country <> 'United States'
  AND country ~* '\m(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WV|WI|WY|DC)\M';

-- United Kingdom variations
UPDATE jobs_cleaned
SET country = 'United Kingdom'
WHERE country ~* '\m(UK|United[[:space:]]+Kingdom|England)\M';

-- France variations
-- France
UPDATE jobs_cleaned
SET country = 'France'
WHERE country ~* '\m(FR|FRANCE)\M';

-- South Africa
UPDATE jobs_cleaned
SET country = 'South Africa'
WHERE country ~* '\m(ZA|SOUTH[[:space:]]+AFRICA)\M';

-- Cyprus
UPDATE jobs_cleaned
SET country = 'Cyprus'
WHERE country ~* '\m(CY|CYPRUS)\M';

-- Ukraine
UPDATE jobs_cleaned
SET country = 'Ukraine'
WHERE country ~* '\m(UA|UKRAINE)\M';

-- Italy
UPDATE jobs_cleaned
SET country = 'Italy'
WHERE country ~* '\m(IT|ITALY)\M';

-- Morocco
UPDATE jobs_cleaned
SET country = 'Morocco'
WHERE country ~* '\m(MAROC|MOROCCO)\M';

-- Czech Republic
UPDATE jobs_cleaned
SET country = 'Czech Republic'
WHERE country ~* '\m(CZECHIA|CZECH[[:space:]]+REPUBLIC)\M';

-- Bosnia and Herzegovina
UPDATE jobs_cleaned
SET country = 'Bosnia and Herzegovina'
WHERE country ~* '\m(BOSNIA|BOSNIA[[:space:]]+AND[[:space:]]+HERZEGOVINA)\M';

-- South Korea
UPDATE jobs_cleaned
SET country = 'South Korea'
WHERE country ~* '\m(KOREA|SOUTH[[:space:]]+KOREA|KOREA,[[:space:]]+REPUBLIC[[:space:]]+OF)\M';

-- Turkey
UPDATE jobs_cleaned
SET country = 'Turkey'
WHERE country ~* '\m(TURKIYE|TURKEY|TR)\M';

-- Tunisia
UPDATE jobs_cleaned
SET country = 'Tunisia'
WHERE country ~* '\m(TUNISIE|TUNISIA)\M';

-- Brazil
UPDATE jobs_cleaned
SET country = 'Brazil'
WHERE country ~* '\m(BRASIL|BRAZIL|BR)\M';

-- Japan
UPDATE jobs_cleaned
SET country = 'Japan'
WHERE country ~* '\m(JP|JAPAN)\M';

-- Uruguay
UPDATE jobs_cleaned
SET country = 'Uruguay'
WHERE country ~* '\m(UY|URUGUAY)\M';

-- Philippines
UPDATE jobs_cleaned
SET country = 'Philippines'
WHERE country ~* '\m(PH|PHILIPPINES)\M';

-- Singapore
UPDATE jobs_cleaned
SET country = 'Singapore'
WHERE country ~* '\m(SG|SINGAPORE)\M';

-- Netherlands
UPDATE jobs_cleaned
SET country = 'Netherlands'
WHERE country ~* '\m(NL|NETHERLANDS|THE[[:space:]]+NETHERLANDS)\M';

-- Sweden
UPDATE jobs_cleaned
SET country = 'Sweden'
WHERE country ~* '\m(SE|SWEDEN)\M';

-- Switzerland
UPDATE jobs_cleaned
SET country = 'Switzerland'
WHERE country ~* '\m(CH|SWITZERLAND)\M';

-- Greece
UPDATE jobs_cleaned
SET country = 'Greece'
WHERE country ~* '\m(GR|GREECE)\M';

-- India
UPDATE jobs_cleaned
SET country = 'India'
WHERE country ~* '\m(IN|INDIA)\M';

-- Portugal
UPDATE jobs_cleaned
SET country = 'Portugal'
WHERE country ~* '\m(PT|PORTUGAL)\M';

-- Australia
UPDATE jobs_cleaned
SET country = 'Australia'
WHERE country ~* '\m(AU|AUSTRALIA)\M';

-- New Zealand
UPDATE jobs_cleaned
SET country = 'New Zealand'
WHERE country ~* '\m(NZ|NEW[[:space:]]+ZEALAND)\M';

-- Belgium
UPDATE jobs_cleaned
SET country = 'Belgium'
WHERE country ~* '\m(BE|BELGIUM)\M';

-- Mexico
UPDATE jobs_cleaned
SET country = 'Mexico'
WHERE country ~* '\m(MX|MEXICO)\M';

-- Russia
UPDATE jobs_cleaned
SET country = 'Russia'
WHERE country ~* '\m(RU|RUSSIA)\M';

-- Egypt
UPDATE jobs_cleaned
SET country = 'Egypt'
WHERE country ~* '\m(EG|EGYPT)\M';

-- Finland
UPDATE jobs_cleaned
SET country = 'Finland'
WHERE country ~* '\m(FI|FINLAND)\M';

-- Saudi Arabia
UPDATE jobs_cleaned
SET country = 'Saudi Arabia'
WHERE country ~* '\m(SA|SAUDI[[:space:]]+ARABIA)\M';

-- Ireland
UPDATE jobs_cleaned
SET country = 'Ireland'
WHERE country ~* '\m(IE|IRELAND)\M';

-- Germany
UPDATE jobs_cleaned
SET country = 'Germany'
WHERE country ~* '\m(DE|GERMANY)\M';

-- Spain
UPDATE jobs_cleaned
SET country = 'Spain'
WHERE country ~* '\m(ES|SPAIN)\M';

-- China
UPDATE jobs_cleaned
SET country = 'China'
WHERE country ~* '\m(CN|CHINA)\M';

-- Canada
UPDATE jobs_cleaned
SET country = 'Canada'
WHERE country ~* '\m(CA|CANADA)\M';

-- United Arab Emirates
UPDATE jobs_cleaned
SET country = 'United Arab Emirates'
WHERE country ~* '\m(UAE|UNITED[[:space:]]+ARAB[[:space:]]+EMIRATES)\M';

-- Continents / regions
UPDATE jobs_cleaned
SET country = 'Europe'
WHERE country ~* '\m(EUROPE)\M';

UPDATE jobs_cleaned
SET country = 'North America'
WHERE country ~* '\m(NORTH[[:space:]]+AMERICA)\M';

UPDATE jobs_cleaned
SET country = 'Latin America'
WHERE country ~* '\m(LATIN[[:space:]]+AMERICA)\M';

UPDATE jobs_cleaned SET country = TRIM(SPLIT_PART(country, ':', 1)) 
WHERE country LIKE '%:%';

UPDATE jobs_cleaned
SET country = TRIM(SPLIT_PART(country, CHR(59), 1))
WHERE country LIKE '%' || CHR(59) || '%';

-- Handle unknown/empty values
UPDATE jobs_cleaned SET country = 'Unknown' WHERE country IS NULL OR TRIM(country) = '' OR UPPER(country) IN ('[ ]', 'UNKNOWN');
 
-- Convert all countries to title case for consistency
UPDATE jobs_cleaned SET country = INITCAP(country);

-- Remove any leading/trailing whitespace that could cause duplicates
UPDATE jobs_cleaned SET country = TRIM(country); 
