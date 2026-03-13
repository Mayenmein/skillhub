-- First, identify and delete duplicates keeping the earliest published job
DELETE FROM jobs_cleaned 
WHERE job_slug IN (
    SELECT job_slug
    FROM (
        SELECT 
            job_slug,
            ROW_NUMBER() OVER (
                PARTITION BY 
                    title, 
                    country, 
                    skills, 
                    salary_min, 
                    salary_max
                ORDER BY published ASC  -- Keep the earliest published version
            ) as rn
        FROM jobs_cleaned
    ) t
    WHERE rn > 1  -- Delete all duplicates (keeping the first one)
);

