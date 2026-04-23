# Compose scenario overlays

Use these files **together with** the root `docker-compose.yml`:

```bash
docker compose -f docker-compose.yml -f compose/docker-compose.redis-password-option-a.yml up -d
```

## Redis credentials: two supported shapes

| File | When to use |
|------|-------------|
| `docker-compose.redis-password-option-a.yml` | **Split credentials**: `REDIS_URL=redis://redis:6379` and `REDIS_PASSWORD=...`. The app injects the password into the DSN at runtime (password never appears in the URL env var). |
| `docker-compose.redis-password-option-b.yml` | **Inline URL**: set `REDIS_URL=redis://:yourpassword@redis:6379/0` and the **same** password in `REDIS_PASSWORD` so the Redis container can run `requirepass`. The app clears `REDIS_PASSWORD` for the `crewai` service so the DSN is not double-encoded. |

Redis must run with `--requirepass` when using either password scenario; the overlay files add that flag.

## News autopilot

`docker-compose.scenario.news-autopilot.yml` adds environment variables to the `crewai` service so the API process periodically ingests an RSS/Atom feed, updates a dataset, and may enqueue retraining when thresholds are met. Copy variables into your own `.env` or merge this file.
