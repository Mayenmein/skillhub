-- Update the original published column to YYYY_MM format
-- Convert timestamp to string and update the column
ALTER TABLE jobs_cleaned 
ALTER COLUMN published TYPE VARCHAR(7) USING TO_CHAR(published, 'YYYY_MM');
