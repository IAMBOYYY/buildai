"""
BuildAI - FastAPI Backend
All API endpoints for the website builder app.
"""

import os
import json
import base64
import httpx
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ai_engine import run_pipeline

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(title="BuildAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────
class BuildRequest(BaseModel):
    prompt: str

class PublishRequest(BaseModel):
    project_id: str
    html_code: str
    subdomain: Optional[str] = None

class GithubSyncRequest(BaseModel):
    project_id: str
    html_code: str
    repo_name: str
    github_token: str

class SupabaseConnectRequest(BaseModel):
    project_id: str
    supabase_url: str
    supabase_anon_key: str


# ── Health check ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gemini_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        "openrouter_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        "supabase_set": bool(os.environ.get("SUPABASE_URL")),
    }


# ── MAIN: Build website (streaming SSE) ───────────────────────────
@app.post("/api/build")
async def build_website(req: BuildRequest):
    """
    Streams the 3-round AI build process to the frontend.
    Uses server-sent events (SSE). Each line is: data: {json}
    """
    if not req.prompt or len(req.prompt.strip()) < 5:
        raise HTTPException(status_code=400, detail="Prompt is too short.")
    if len(req.prompt) > 2000:
        raise HTTPException(status_code=400, detail="Prompt is too long (max 2000 characters).")

    return StreamingResponse(
        run_pipeline(req.prompt.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ── Publish website ────────────────────────────────────────────────
@app.post("/api/publish")
async def publish_website(req: PublishRequest):
    """
    Publishes a website. Saves code to Supabase if configured.
    Returns the live URL.
    """
    subdomain = req.subdomain or f"project-{req.project_id[:8]}"
    # Remove any characters that are not url-safe
    subdomain = "".join(c for c in subdomain.lower() if c.isalnum() or c == "-")[:30]
    published_url = f"https://{subdomain}.buildai.app"

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

    if supabase_url and supabase_key:
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.patch(
                f"{supabase_url}/rest/v1/projects?id=eq.{req.project_id}",
                json={
                    "published": True,
                    "published_url": published_url,
                    "html_code": req.html_code,
                    "published_at": datetime.now(timezone.utc).isoformat()
                },
                headers=headers
            )

    return {
        "success": True,
        "url": published_url,
        "subdomain": subdomain,
        "message": "Website published successfully!"
    }


# ── GitHub sync ────────────────────────────────────────────────────
@app.post("/api/github/sync")
async def github_sync(req: GithubSyncRequest):
    """Push website HTML to a GitHub repository."""
    headers = {
        "Authorization": f"Bearer {req.github_token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get authenticated user
        user_resp = await client.get("https://api.github.com/user", headers=headers)
        if user_resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid GitHub token.")
        github_user = user_resp.json()["login"]
        repo_full = f"{github_user}/{req.repo_name}"

        # Create repo if it does not exist
        repo_resp = await client.get(f"https://api.github.com/repos/{repo_full}", headers=headers)
        if repo_resp.status_code == 404:
            create_resp = await client.post(
                "https://api.github.com/user/repos",
                json={
                    "name": req.repo_name,
                    "description": "Built with BuildAI — AI Website Builder",
                    "private": False,
                    "auto_init": True
                },
                headers=headers
            )
            if create_resp.status_code not in (200, 201):
                raise HTTPException(status_code=500, detail=f"Could not create repo: {create_resp.text}")

        # Get existing file SHA for update (if file already exists)
        file_resp = await client.get(
            f"https://api.github.com/repos/{repo_full}/contents/index.html",
            headers=headers
        )
        sha = file_resp.json().get("sha") if file_resp.status_code == 200 else None

        # Push index.html
        content_b64 = base64.b64encode(req.html_code.encode()).decode()
        push_body = {
            "message": f"Update via BuildAI — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            "content": content_b64,
        }
        if sha:
            push_body["sha"] = sha

        push_resp = await client.put(
            f"https://api.github.com/repos/{repo_full}/contents/index.html",
            json=push_body,
            headers=headers
        )

        if push_resp.status_code in (200, 201):
            return {
                "success": True,
                "repo_url": f"https://github.com/{repo_full}",
                "pages_url": f"https://{github_user}.github.io/{req.repo_name}",
                "message": "Pushed to GitHub successfully!"
            }
        raise HTTPException(status_code=500, detail=f"GitHub push failed: {push_resp.text}")


# ── Supabase connection validator ─────────────────────────────────
@app.post("/api/supabase/connect")
async def connect_supabase(req: SupabaseConnectRequest):
    """Validate Supabase credentials by making a test request."""
    headers = {
        "apikey": req.supabase_anon_key,
        "Authorization": f"Bearer {req.supabase_anon_key}",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{req.supabase_url}/rest/v1/", headers=headers)
        if resp.status_code in (200, 400):
            return {
                "success": True,
                "message": "Supabase connected successfully!",
                "features": ["auth", "database", "storage", "edge_functions"]
            }
        raise HTTPException(
            status_code=400,
            detail="Could not connect to Supabase. Check your URL and anon key."
        )


# ── Serve the frontend ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the main app HTML."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>index.html not found</h1><p>Make sure index.html is in the same folder as main.py</p>",
            status_code=404
        )


@app.get("/{full_path:path}", response_class=HTMLResponse)
async def catch_all(full_path: str):
    """Catch-all: always serve index.html for SPA routing."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="Not found", status_code=404)
