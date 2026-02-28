-- Create a cleaning table for data science job market analysis
DROP TABLE IF EXISTS jobs_cleaned;
CREATE TABLE jobs_cleaned AS
SELECT 
    -- Core job identifiers
    j.slug AS job_slug,
    j.title, 
     
    j.skills,
    
    -- Job details
    j.seniority,
    j.types,  -- Employment types array
    
    -- Location
    j.country,
    j.city,
    
    -- Salary data (for SQL cleaning)
    j.salary_min,
    j.salary_max,
    j.salary_currency,
    j.salary_period,
    
    -- Dates (for SQL cleaning)
    j.published,
    
    -- Company information
    c.name AS company_name,
    c.company_sector  
     
    
FROM jobs j
LEFT JOIN companies c ON j.company_slug = c.slug
WHERE j.ai = TRUE;  

-- Create indexes on frequently filtered columns
CREATE INDEX idx_cleaned_published ON jobs_cleaned(published);
CREATE INDEX idx_cleaned_country ON jobs_cleaned(country);
CREATE INDEX idx_cleaned_seniority ON jobs_cleaned(seniority);
CREATE INDEX idx_cleaned_company_sector ON jobs_cleaned(company_sector);
 
CREATE INDEX idx_cleaned_skills ON jobs_cleaned USING GIN(skills); 