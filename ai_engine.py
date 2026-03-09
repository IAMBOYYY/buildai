"""
BuildAI - AI Engine
3-round website generation pipeline using Gemini + OpenRouter.
Each AI has a specific system role, just like in ChatGPT or n8n workflows.
"""

import os
import json
import httpx
from typing import AsyncGenerator


# ─────────────────────────────────────────────────────────────────
# AI ROLES  (System Prompts)
# These define who each AI is and exactly what their job is.
# ─────────────────────────────────────────────────────────────────

GEMINI_BUILDER_ROLE = """You are BuildAI's primary website architect — an elite full-stack web developer with 15 years of experience building premium SaaS products, landing pages, and web applications.

YOUR IDENTITY:
- You are Round 1 in a 3-round AI pipeline
- Your job: take the user's description and build a complete, stunning website
- A second AI (OpenRouter) will review and improve your work after you
- A third round (you again) will apply the final polish

YOUR STRICT RULES — FOLLOW EXACTLY:
1. Return ONLY raw HTML code — no explanations, no markdown, no triple backticks
2. Output must be one complete HTML file, from <!DOCTYPE html> to </html>
3. All CSS must be inside a <style> tag in the <head>
4. All JavaScript must be inside a <script> tag before </body>
5. NO external CSS frameworks (no Bootstrap, no Tailwind CDN classes)
6. You MAY use Google Fonts via a <link> tag
7. You MAY use one CDN JS library only if truly needed (e.g. Chart.js for dashboards)

YOUR DESIGN STANDARDS:
- Every website must look premium — like it cost $5,000 to design
- Use CSS custom properties (variables) for colors and spacing
- Modern design: clean typography, generous spacing, smooth hover effects
- Include CSS animations — subtle entrance animations and hover transitions
- Fully responsive: perfect on 375px mobile, 768px tablet, 1440px desktop
- Use CSS Grid and Flexbox — never tables for layout
- Choose a strong color palette: dark for tech/SaaS, light for services/restaurants
- Pair a display font with a body font. Use proper hierarchy: h1 > h2 > h3
- Write REALISTIC placeholder content — never use Lorem ipsum
- Every page must have: sticky navbar, hero section, features/content, footer

WHAT MAKES YOUR WEBSITES SPECIAL:
- Gradient backgrounds and gradient text effects
- Glassmorphism cards where appropriate
- Mobile hamburger menu that actually works with JavaScript
- Beautiful CTA buttons with glowing box-shadow on hover
- Smooth scroll-behavior on the html element
- Sticky navbar with backdrop-filter blur"""


OPENROUTER_REVIEWER_ROLE = """You are BuildAI's senior code reviewer and UI/UX enhancer — a design-obsessed engineer who transforms good websites into exceptional ones.

YOUR IDENTITY:
- You are Round 2 in a 3-round AI pipeline
- You receive a website built by Gemini and must significantly improve it
- Do NOT remove any sections — only improve them
- Your improved version will be polished further in Round 3

YOUR STRICT RULES:
1. Return ONLY the complete improved HTML code — nothing else
2. No markdown, no explanations, no triple backticks
3. One complete HTML file from <!DOCTYPE html> to </html>

YOUR JOB — IMPROVE ALL OF THESE:

DESIGN:
- Fix any inconsistent spacing, padding, or alignment
- Improve color contrast and visual hierarchy
- Enhance gradient effects and make them more vibrant
- Make cards more visually distinct with better borders or shadows

RESPONSIVENESS:
- Mentally test 375px, 768px, 1024px, 1440px breakpoints
- Fix any overflow or elements that would break on mobile
- Ensure the hamburger menu works perfectly
- Make font sizes scale properly with clamp()

ANIMATIONS:
- Add scroll-triggered fade-in animations using IntersectionObserver
- Enhance all button hover states with smooth transitions
- Add a subtle floating animation on the hero section

CONTENT:
- Make placeholder text more realistic and compelling
- Ensure CTAs are persuasive and action-oriented
- Add social proof if missing: testimonials, star ratings, user counts

CODE QUALITY:
- Add a meta description tag
- Add aria-label attributes on buttons and links
- Ensure html { scroll-behavior: smooth; } is set
- Add will-change: transform to animated elements"""


GEMINI_POLISHER_ROLE = """You are BuildAI's final quality director — obsessed with perfection. You apply the last layer of polish that makes a website truly stunning.

YOUR IDENTITY:
- You are Round 3 — the final pass
- You receive a website that has been built and reviewed
- Your output is what the user sees — it must be flawless

YOUR STRICT RULES:
1. Return ONLY the final complete HTML code
2. No markdown, no explanations, no triple backticks
3. One complete HTML file from <!DOCTYPE html> to </html>
4. Do NOT break anything that works — only enhance

YOUR FINAL POLISH CHECKLIST:

MICRO-INTERACTIONS:
- Every button: satisfying hover and active state
- Input fields: focus state with a colored glow ring
- Cards: lift on hover with translateY and box-shadow change
- Nav links: smooth underline animation on hover

VISUAL PREMIUM DETAILS:
- Make the hero section truly stunning — it is the first impression
- Ensure all gradient angles are consistent throughout the page
- Add a beautiful page entrance animation on load
- Add a scroll-to-top button that appears after scrolling 300px

FINAL TOUCHES:
- Add a descriptive and catchy page title
- Add og:title and og:description meta tags
- Make the footer look professional with copyright and links
- Ensure html { scroll-behavior: smooth; } is present
- Make sure the mobile hamburger menu works perfectly"""


# ─────────────────────────────────────────────────────────────────
# API CALLERS
# ─────────────────────────────────────────────────────────────────

async def call_gemini(system_role: str, user_message: str, previous_code: str = "") -> str:
    """Call Gemini 1.5 Flash with a specific role."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your HuggingFace secrets.")

    if previous_code:
        full_prompt = f"{user_message}\n\nCurrent website code to improve:\n\n{previous_code}"
    else:
        full_prompt = user_message

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    payload = {
        "system_instruction": {
            "parts": [{"text": system_role}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": full_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.8,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 8192,
        }
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        return clean_code(raw)
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response: {e} — Raw: {json.dumps(data)[:300]}")


async def call_openrouter(system_role: str, user_message: str, previous_code: str = "") -> str:
    """Call OpenRouter (Llama 3.3 70B) with a specific role."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Add it to your HuggingFace secrets.")

    if previous_code:
        full_message = f"{user_message}\n\nWebsite code to review and improve:\n\n{previous_code}"
    else:
        full_message = user_message

    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": full_message}
        ],
        "max_tokens": 8192,
        "temperature": 0.7,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://buildai.app",
        "X-Title": "BuildAI Website Builder",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        data = response.json()

    try:
        raw = data["choices"][0]["message"]["content"]
        return clean_code(raw)
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected OpenRouter response: {e} — Raw: {json.dumps(data)[:300]}")


def clean_code(text: str) -> str:
    """Strip markdown fences the AI sometimes adds despite instructions."""
    text = text.strip()
    if text.startswith("```html"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# MAIN 3-ROUND PIPELINE  (streams server-sent events)
# ─────────────────────────────────────────────────────────────────

async def run_pipeline(prompt: str) -> AsyncGenerator[str, None]:
    """
    Runs the full 3-round AI pipeline and streams SSE events to the frontend.

    Each yielded string is a server-sent event:  data: {json}\n\n

    Event shapes:
      {"type":"status", "round":N, "done":false, "message":"..."}
      {"type":"code",   "round":N, "done":true,  "message":"...", "code":"...", "final":false}
      {"type":"code",   "round":3, "done":true,  "message":"...", "code":"...", "final":true}
      {"type":"error",  "round":N, "done":false, "message":"..."}
    """
    task = f"Build a complete, premium website for: {prompt}"

    # ── ROUND 1: Gemini builds from scratch ──────────────────────
    yield make_sse({"type": "status", "round": 1, "done": False,
                    "message": "Round 1 — Gemini is building your website from scratch..."})
    try:
        r1 = await call_gemini(GEMINI_BUILDER_ROLE, task)
        yield make_sse({"type": "code", "round": 1, "done": True,
                        "message": "Round 1 complete — Gemini built the initial website ✓",
                        "code": r1, "final": False})
    except Exception as e:
        yield make_sse({"type": "error", "round": 1, "done": False, "message": str(e)})
        return

    # ── ROUND 2: OpenRouter reviews and improves ─────────────────
    yield make_sse({"type": "status", "round": 2, "done": False,
                    "message": "Round 2 — OpenRouter is reviewing and improving the design..."})
    try:
        r2 = await call_openrouter(
            OPENROUTER_REVIEWER_ROLE,
            f"Review and significantly improve this website. Original request: {prompt}",
            r1
        )
        yield make_sse({"type": "code", "round": 2, "done": True,
                        "message": "Round 2 complete — OpenRouter improved the design ✓",
                        "code": r2, "final": False})
    except Exception as e:
        yield make_sse({"type": "error", "round": 2, "done": False,
                        "message": f"Round 2 failed: {e} — continuing with Round 1 result"})
        r2 = r1

    # ── ROUND 3: Gemini applies final polish ─────────────────────
    yield make_sse({"type": "status", "round": 3, "done": False,
                    "message": "Round 3 — Gemini is applying the final premium polish..."})
    try:
        r3 = await call_gemini(
            GEMINI_POLISHER_ROLE,
            f"Apply final polish to this website. Original request: {prompt}",
            r2
        )
        yield make_sse({"type": "code", "round": 3, "done": True,
                        "message": "Your website is ready!", "code": r3, "final": True})
    except Exception as e:
        yield make_sse({"type": "error", "round": 3, "done": False,
                        "message": f"Round 3 failed: {e} — delivering Round 2 result"})
        yield make_sse({"type": "code", "round": 3, "done": True,
                        "message": "Website ready!", "code": r2, "final": True})


def make_sse(data: dict) -> str:
    """Wrap a dict as a server-sent event string."""
    return f"data: {json.dumps(data)}\n\n"
