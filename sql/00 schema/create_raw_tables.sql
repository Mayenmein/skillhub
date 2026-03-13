-- Drop tables if they exist (in reverse order of dependencies)
DROP TABLE IF EXISTS job_skills_detail CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;
DROP TABLE IF EXISTS companies CASCADE;

-- Create companies table
CREATE TABLE companies (
    -- Core identifiers
    id SERIAL PRIMARY KEY,
    slug VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    
    -- Company details
    description TEXT,
    description_gpt TEXT,
    description_premium TEXT,
    description_linkedin TEXT,
    description_combined_gpt TEXT,
    description_perplexity TEXT,
    logo VARCHAR(500),
    url VARCHAR(500),
    
    -- Location
    city VARCHAR(255),
    country VARCHAR(50),
    areas TEXT,
    
    -- Company metrics
    size_min INTEGER,
    size_max INTEGER,
    year_founded INTEGER,
    glassdoor_score DECIMAL(3,1),
    linkedin_staff_count INTEGER,
    linkedin_follower_count INTEGER,
    
    -- Social media
    twitter VARCHAR(255),
    linkedin VARCHAR(500),
    facebook VARCHAR(255),
    instagram VARCHAR(255),
    github VARCHAR(255),
    crunchbase VARCHAR(255),
    angel_list VARCHAR(255),
    stack_overflow VARCHAR(255),
    behance VARCHAR(255),
    dribbble VARCHAR(255),
    product_hunt VARCHAR(255),
    
    -- Company classification
    public_vs_private VARCHAR(100),
    company_sector TEXT,
    linkedin_tags TEXT,
    
    -- Products and AI
    key_products TEXT,
    using_ai TEXT,
    ai_products_gpt TEXT,
    
    -- Mission and values
    mission_values TEXT,
    
    -- Job statistics
    jobs_count INTEGER DEFAULT 0,
    jobs_ai_count INTEGER DEFAULT 0,
    last_job_source VARCHAR(255),
    
    -- Metadata
    created_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create jobs table
CREATE TABLE jobs (
    -- Core identifiers
    id SERIAL PRIMARY KEY,
    slug VARCHAR(255) UNIQUE NOT NULL,
    company_slug VARCHAR(255) REFERENCES companies(slug) ON DELETE CASCADE,
    
    -- Job details
    title TEXT NOT NULL,
    description TEXT,
    premium BOOLEAN DEFAULT FALSE,
    ai BOOLEAN DEFAULT FALSE,
    status INTEGER,
    created_at TIMESTAMP WITH TIME ZONE,
    published TIMESTAMP WITH TIME ZONE,
    pin_until TIMESTAMP WITH TIME ZONE,
    
    -- Skills and requirements (stored as arrays for easier querying)
    skills TEXT[],
    soft_skills TEXT[],
    tools TEXT[],
    languages TEXT[],
    frameworks TEXT[],
    libraries TEXT[],
    roles TEXT,
    seniority VARCHAR(100),
    
    -- Job type and location
    types TEXT[], -- Array of employment types
    city VARCHAR(255),
    country VARCHAR(255),
    location TEXT, -- Combined location string
    
    -- Compensation
    salary_min INTEGER DEFAULT 0,
    salary_max INTEGER DEFAULT 0,
    salary_currency VARCHAR(10),
    salary_period VARCHAR(50),
    
    -- Additional fields
    benefits TEXT,
    beneficial TEXT,
    experience TEXT,
    ideal_candidate TEXT,
    qualifications TEXT,
    schema TEXT,
    force_status VARCHAR(50),
    same_as INTEGER DEFAULT 0,
    
    -- Metadata
    created_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create job_skills_detail table for normalized skill tracking
CREATE TABLE job_skills_detail (
    id SERIAL PRIMARY KEY,
    job_slug VARCHAR(255) REFERENCES jobs(slug) ON DELETE CASCADE,
    skill_name VARCHAR(255) NOT NULL,
    skill_category VARCHAR(50), -- 'technical', 'soft', 'language', 'tool', 'framework', 'library'
    created_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure unique skill per job
    UNIQUE(job_slug, skill_name, skill_category)
);

-- Create indexes for companies table
CREATE INDEX idx_companies_name ON companies(name);
CREATE INDEX idx_companies_slug ON companies(slug);
CREATE INDEX idx_companies_country ON companies(country);
CREATE INDEX idx_companies_city ON companies(city);
CREATE INDEX idx_companies_sector ON companies(company_sector);
CREATE INDEX idx_companies_year_founded ON companies(year_founded);
CREATE INDEX idx_companies_linkedin_staff ON companies(linkedin_staff_count);
CREATE INDEX idx_companies_jobs_count ON companies(jobs_count);
CREATE INDEX idx_companies_ai_jobs_count ON companies(jobs_ai_count);

-- Create indexes for jobs table
CREATE INDEX idx_jobs_slug ON jobs(slug);
CREATE INDEX idx_jobs_company_slug ON jobs(company_slug);
CREATE INDEX idx_jobs_title ON jobs(title);
CREATE INDEX idx_jobs_published ON jobs(published DESC);
CREATE INDEX idx_jobs_created ON jobs(created_at_utc DESC);
CREATE INDEX idx_jobs_country ON jobs(country);
CREATE INDEX idx_jobs_city ON jobs(city);
CREATE INDEX idx_jobs_ai ON jobs(ai);
CREATE INDEX idx_jobs_premium ON jobs(premium);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_seniority ON jobs(seniority);
CREATE INDEX idx_jobs_salary_min ON jobs(salary_min);
CREATE INDEX idx_jobs_salary_max ON jobs(salary_max);

-- Create GIN indexes for array columns (for faster array operations)
CREATE INDEX idx_jobs_skills_gin ON jobs USING GIN(skills);
CREATE INDEX idx_jobs_soft_skills_gin ON jobs USING GIN(soft_skills);
CREATE INDEX idx_jobs_tools_gin ON jobs USING GIN(tools);
CREATE INDEX idx_jobs_languages_gin ON jobs USING GIN(languages);
CREATE INDEX idx_jobs_frameworks_gin ON jobs USING GIN(frameworks);
CREATE INDEX idx_jobs_libraries_gin ON jobs USING GIN(libraries);
CREATE INDEX idx_jobs_types_gin ON jobs USING GIN(types);

-- Create composite indexes for common query patterns
CREATE INDEX idx_jobs_country_ai ON jobs(country, ai) WHERE ai = true;
CREATE INDEX idx_jobs_published_ai ON jobs(published, ai) WHERE ai = true;
CREATE INDEX idx_jobs_company_published ON jobs(company_slug, published DESC);
CREATE INDEX idx_jobs_location_skills ON jobs(country, city) INCLUDE (skills);

-- Create indexes for job_skills_detail table
CREATE INDEX idx_skill_detail_job_slug ON job_skills_detail(job_slug);
CREATE INDEX idx_skill_detail_name ON job_skills_detail(skill_name);
CREATE INDEX idx_skill_detail_category ON job_skills_detail(skill_category);
CREATE INDEX idx_skill_detail_name_category ON job_skills_detail(skill_name, skill_category);
 
-- Create partial indexes for specific use cases
--CREATE INDEX idx_jobs_recent_active ON jobs(published) 
  --  WHERE published > CURRENT_TIMESTAMP - INTERVAL '30 days';
--CREATE INDEX idx_jobs_with_salary ON jobs(salary_min, salary_max) 
  --  WHERE salary_min > 0 OR salary_max > 0;

-- Create function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at_utc = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_companies_updated_at 
    BEFORE UPDATE ON companies 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at 
    BEFORE UPDATE ON jobs 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create view for job listings with company details (useful for frequent queries)
CREATE VIEW job_listings_with_company AS
SELECT 
    j.*,
    c.name AS company_name,
    c.logo AS company_logo,
    c.url AS company_url,
    c.description AS company_description,
    c.linkedin_staff_count AS company_size,
    c.year_founded AS company_year_founded,
    c.company_sector,
    c.linkedin_follower_count
FROM jobs j
LEFT JOIN companies c ON j.company_slug = c.slug;

-- Create function to update company job counts
CREATE OR REPLACE FUNCTION update_company_job_counts()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Increment job counts
        UPDATE companies 
        SET 
            jobs_count = jobs_count + 1,
            jobs_ai_count = jobs_ai_count + CASE WHEN NEW.ai THEN 1 ELSE 0 END
        WHERE slug = NEW.company_slug;
    ELSIF TG_OP = 'DELETE' THEN
        -- Decrement job counts
        UPDATE companies 
        SET 
            jobs_count = jobs_count - 1,
            jobs_ai_count = jobs_ai_count - CASE WHEN OLD.ai THEN 1 ELSE 0 END
        WHERE slug = OLD.company_slug;
    ELSIF TG_OP = 'UPDATE' AND OLD.ai != NEW.ai THEN
        -- Adjust ai count if ai status changed
        UPDATE companies 
        SET 
            jobs_ai_count = jobs_ai_count + CASE WHEN NEW.ai THEN 1 ELSE -1 END
        WHERE slug = NEW.company_slug;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Create triggers to maintain company job counts
CREATE TRIGGER maintain_company_job_counts_insert
    AFTER INSERT ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_company_job_counts();

CREATE TRIGGER maintain_company_job_counts_delete
    AFTER DELETE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_company_job_counts();

CREATE TRIGGER maintain_company_job_counts_update
    AFTER UPDATE OF ai ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_company_job_counts();

-- Create comments for documentation
COMMENT ON TABLE companies IS 'Companies that post job listings';
COMMENT ON TABLE jobs IS 'Job listings from various companies';
COMMENT ON TABLE job_skills_detail IS 'Normalized skills extracted from job listings';

COMMENT ON COLUMN jobs.skills IS 'Array of technical skills required for the job';
COMMENT ON COLUMN jobs.types IS 'Array of employment types (Remote, Full Time, Part Time, Freelancer)';
COMMENT ON COLUMN jobs.ai IS 'Indicates if this is an AI-related role';
COMMENT ON COLUMN companies.linkedin_staff_count IS 'Number of employees according to LinkedIn';
COMMENT ON COLUMN companies.linkedin_follower_count IS 'Number of LinkedIn followers';