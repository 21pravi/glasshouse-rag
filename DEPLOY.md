# Deploying the online engine

The online engine works by having a server hold the API key and call Anthropic
server-side. The browser only ever sends `{query, mode}` — no key crosses the
network to the client, and there is no CORS involved. This is the setup the live
demo uses.

## Render (one blueprint)

1. Push this repo to GitHub (it already contains `render.yaml`).
2. In Render: **New +** → **Blueprint** → select the repo → **Apply**.
3. Open the service → **Environment** → add:
   `ANTHROPIC_API_KEY = sk-ant-...`
4. Render redeploys. Visit the service URL.

`render.yaml` pins Python 3.11, installs `requirements-online.txt`, starts
`uvicorn server:app`, and points the health check at `/healthz`. Leaving
`ANTHROPIC_API_KEY` unset is valid — the service runs offline-only.

Cost note: on Render's free tier the service sleeps after inactivity and takes
~30 s to wake on the next request. Fine for a portfolio demo.

## Any other host

The app is a standard ASGI application. Anywhere that runs

    uvicorn server:app --host 0.0.0.0 --port $PORT

will work. `Procfile` covers Railway / Heroku-style platforms; `runtime.txt`
pins the Python version for buildpacks. Set `ANTHROPIC_API_KEY` (or
`OPENAI_API_KEY` for gpt-4o-mini) in the platform's secret manager — never in
git.

## Verifying a deployment

```bash
curl https://YOUR-APP/healthz
# {"status":"ok"}

curl https://YOUR-APP/api/health
# {"status":"ok","online_available":true,"online_provider":"claude"}

curl -X POST https://YOUR-APP/api/ask \
  -H 'content-type: application/json' \
  -d '{"query":"How do I reset my VPN password?","mode":"both"}'
# offline: the verbatim tech_faqs.md sentence
# online : a generated answer
# snippets: tech_faqs.md at ~7.57
```

If `online_available` is `false`, the key is not set in the environment. If the
offline answer is correct but online is `null` with an error, the key is set but
rejected — check it in the dashboard.

## Two deployments, one repo

GitHub Pages (serving `docs/`) and the hosted server are not mutually exclusive.
A common setup: Pages for a permanently-free offline demo linked from the README,
and a Render URL for the full offline-plus-online comparison. Point recruiters at
whichever you prefer; both run the same retrieval and the same offline engine.
