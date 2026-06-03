import os
import base64
import hashlib
import secrets
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SCOPES = "user-read-private playlist-modify-public playlist-modify-private"

# --- PKCE helpers ---
def generate_code_verifier():
    return secrets.token_urlsafe(64)

def generate_code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

# --- Local callback server ---
auth_code_holder = {"code": None}

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            auth_code_holder["code"] = params["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<h2>Auth successful! You can close this tab.</h2>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h2>Auth failed. No code received.</h2>")

    def log_message(self, format, *args):
        pass  # Suppress server logs

def get_auth_code(verifier: str) -> str:
    challenge = generate_code_challenge(verifier)
    state = secrets.token_urlsafe(16)

    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    auth_url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)

    server = HTTPServer(("127.0.0.1", 5000), CallbackHandler)
    thread = Thread(target=server.handle_request)
    thread.start()

    print("\n🎵 Opening Spotify login in your browser...")
    webbrowser.open(auth_url)
    thread.join(timeout=60)

    code = auth_code_holder.get("code")
    if not code:
        raise RuntimeError("No auth code received. Did you authorize in the browser?")
    return code

def exchange_code_for_token(code: str, verifier: str) -> dict:
    credentials = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": verifier,
        },
    )
    response.raise_for_status()
    return response.json()

def get_access_token() -> str:
    verifier = generate_code_verifier()
    code = get_auth_code(verifier)
    tokens = exchange_code_for_token(code, verifier)
    print("✅ Auth successful!\n")
    return tokens["access_token"]