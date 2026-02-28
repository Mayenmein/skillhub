CREATE TABLE datajobscompanies (
    id INTEGER PRIMARY KEY,
    source_id INTEGER,
    name VARCHAR(255),
    logo VARCHAR(500),
    website_url TEXT,
    linkedin_url TEXT,
    twitter_handle VARCHAR(255),
    github_url TEXT,
    is_agency BOOLEAN
);  


CREATE TABLE datajobs (
    id INTEGER PRIMARY KEY,
    ext_id VARCHAR(255),
    company_id INTEGER REFERENCES datajobscompanies(id),
    title TEXT,
    location TEXT,
    description TEXT,
    experience_level VARCHAR(10),
    application_url TEXT,
    language VARCHAR(10),
    has_remote BOOLEAN,
    published TIMESTAMP WITH TIME ZONE,
    salary_min INTEGER,
    salary_max INTEGER,
    salary_currency VARCHAR(3),
    created_at_utc TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);  


CREATE TABLE datajob_types (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE datajob_to_types (
    job_id INTEGER REFERENCES datajobs(id),
    type_id INTEGER REFERENCES datajob_types(id),
    PRIMARY KEY (job_id, type_id)
); 

CREATE TABLE datajobslocations (
    geonameid INTEGER PRIMARY KEY,
    asciiname VARCHAR(255),
    name VARCHAR(255),
    country_code VARCHAR(2),
    state_code VARCHAR(50),
    timezone VARCHAR(100),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    population BIGINT
);

CREATE TABLE datajob_locations (
    job_id INTEGER REFERENCES datajobs(id),
    geonameid INTEGER REFERENCES datajobslocations(geonameid),
    PRIMARY KEY (job_id, geonameid)
); 

CREATE TABLE datajobscountries (
    id INTEGER PRIMARY KEY,
    code VARCHAR(2),
    name VARCHAR(255),
    region_id INTEGER,
    region_name VARCHAR(100)
);

-- Add indexes for foreign keys and commonly queried columns
 
CREATE INDEX idx_datajobscompanies_source_id ON datajobscompanies(source_id);
 
CREATE INDEX idx_datajobscountries_region_id ON datajobscountries(region_id);
 
CREATE INDEX idx_datajobs_ext_id ON datajobs(ext_id);
CREATE INDEX idx_datajobs_published ON datajobs(published);
CREATE INDEX idx_datajobs_experience_level ON datajobs(experience_level);
CREATE INDEX idx_datajobs_has_remote ON datajobs(has_remote);
 
CREATE INDEX idx_datajobslocations_country_code ON datajobslocations(country_code);
CREATE INDEX idx_datajobslocations_state_code ON datajobslocations(state_code);