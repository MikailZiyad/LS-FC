import base64
import json
import os
from pathlib import Path
from typing import List, Optional
import urllib.request
import urllib.error


def repo_owner_repo_from_url(url: str) -> tuple[str, str]:
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]
    return owner, repo


def http_request(method: str, url: str, token: str, data: Optional[dict] = None) -> tuple[int, dict]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {token}",
        "User-Agent": "trae-uploader",
    }
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.getcode()
            text = resp.read().decode("utf-8")
            return status, json.loads(text) if text else {}
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            text = e.read().decode("utf-8")
            obj = json.loads(text) if text else {}
        except Exception:
            obj = {"message": str(e)}
        return status, obj


def load_ignore_patterns(gitignore_path: Path) -> List[str]:
    patterns: List[str] = []
    if not gitignore_path.exists():
        return patterns
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def path_matches_patterns(rel: str, patterns: List[str]) -> bool:
    from fnmatch import fnmatch
    # normalize to posix style
    r = rel.replace("\\", "/")
    for p in patterns:
        p2 = p.replace("\\", "/")
        # Ensure trailing slash patterns match directories and their contents
        if p2.endswith("/"):
            if r == p2[:-1] or r.startswith(p2):
                return True
        if fnmatch(r, p2) or fnmatch(r.split("/")[-1], p2):
            return True
    return False


def get_default_branch(owner: str, repo: str, token: str) -> str:
    status, data = http_request("GET", f"https://api.github.com/repos/{owner}/{repo}", token)
    if status == 200:
        return data.get("default_branch", "main")
    # Fallback to main
    return "main"


def get_remote_sha(owner: str, repo: str, path: str, token: str, branch: str) -> Optional[str]:
    status, data = http_request(
        "GET",
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}",
        token,
    )
    if status == 200 and isinstance(data, dict):
        return data.get("sha")
    return None


def put_file(owner: str, repo: str, path: str, content_bytes: bytes, token: str, branch: str, sha: Optional[str]) -> bool:
    b64 = base64.b64encode(content_bytes).decode("utf-8")
    payload = {
        "message": f"chore: add {path}",
        "content": b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
        payload["message"] = f"chore: update {path}"
    status, data = http_request(
        "PUT",
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
        token,
        data=payload,
    )
    if status in (200, 201):
        return True
    # Print concise error without secrets
    msg = data.get("message", f"HTTP {status}")
    print(f"Failed to upload {path}: {msg}")
    return False


def main():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo_url = os.environ.get("GITHUB_REPO_URL", "").strip()  # e.g. https://github.com/owner/repo.git

    if not token:
        print("ERROR: GITHUB_TOKEN is not set.")
        raise SystemExit(1)
    if not repo_url:
        print("ERROR: GITHUB_REPO_URL is not set.")
        raise SystemExit(1)

    owner, repo = repo_owner_repo_from_url(repo_url)

    repo_root = Path(__file__).resolve().parents[1]
    gitignore_path = repo_root / ".gitignore"
    patterns = load_ignore_patterns(gitignore_path)

    default_branch = get_default_branch(owner, repo, token)
    print(f"Uploading to {owner}/{repo} on branch {default_branch}")

    # Safety: skip files larger than ~95 MB (GitHub limit is 100 MB)
    MAX_SIZE = 95 * 1024 * 1024

    uploaded = 0
    skipped = 0
    for p in repo_root.rglob("*"):
        if p.is_dir():
            # Skip .git directory explicitly
            if p.name == ".git":
                continue
            # Ignore directories via .gitignore patterns
            rel_dir = p.relative_to(repo_root).as_posix()
            if path_matches_patterns(rel_dir + "/", patterns):
                continue
            continue

        rel_path = p.relative_to(repo_root).as_posix()
        if path_matches_patterns(rel_path, patterns):
            skipped += 1
            continue
        if p.stat().st_size > MAX_SIZE:
            print(f"Skip large file (>95MB): {rel_path}")
            skipped += 1
            continue

        # Read file content in binary
        try:
            data = p.read_bytes()
        except Exception as e:
            print(f"Skip unreadable file {rel_path}: {e}")
            skipped += 1
            continue

        # Determine existing sha (if file already exists)
        sha = get_remote_sha(owner, repo, rel_path, token, default_branch)
        ok = put_file(owner, repo, rel_path, data, token, default_branch, sha)
        if ok:
            uploaded += 1
        else:
            skipped += 1

    print(f"Done. Uploaded: {uploaded}, skipped: {skipped}")


if __name__ == "__main__":
    main()

