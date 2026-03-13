-- Remove generic skills from skills array
UPDATE jobs_cleaned 
SET skills = ARRAY(
    SELECT skill
    FROM unnest(skills) AS skill
    WHERE skill NOT IN (
        'Data Science',
        'Artificial Intelligence',
        'AI',
        'Analytics',
        'Data Analysis',
        'Programming',
        'Coding',
        'Statistics',
        'ML',
        'Machine Learning',
        'Data Engineer',
        'Data Engineering',
        'Data',
        'Science',
        'Business Intelligence',
        'BI',
        'Computer Science',
        'CS',
        'IT',
        'Information Technology',
        'Software Development',
        'Development',
        'Engineering',
        'Research',
        'R&D'
    )
)
WHERE skills IS NOT NULL;
