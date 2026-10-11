-- Read-only role for the web admin. Password is set out-of-band:
--   ALTER ROLE admin_ro PASSWORD '...';
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'admin_ro') THEN
        CREATE ROLE admin_ro LOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO admin_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO admin_ro;
-- Tables created later by the migrating role are readable too.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO admin_ro;
