-- Read-only role for the web admin. Password is set out-of-band:
--   ALTER ROLE admin_ro PASSWORD '...';
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'admin_ro') THEN
        BEGIN
            CREATE ROLE admin_ro LOGIN;
        EXCEPTION WHEN duplicate_object OR unique_violation THEN
            NULL; -- concurrent migration created it first (roles are cluster-global)
        END;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO admin_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO admin_ro;
-- Covers only tables created later by the migrating role (not other roles).
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO admin_ro;
