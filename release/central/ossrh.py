#!/usr/bin/env python3
"""Explicit OPEN OSSRH compatibility staging. No close, release, or Portal upload API.

See OPEN-STAGING.md for API sources and the bootstrap/upload operator contract.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.request import Request, build_opener, HTTPRedirectHandler

try:
    from .repository import sign_repository, resolve_maven_command
    from .verify_java_snapshots import inventory
except ImportError:  # direct Actions invocation
    from repository import sign_repository, resolve_maven_command
    from verify_java_snapshots import inventory

ENDPOINT = "https://ossrh-staging-api.central.sonatype.com"
PLUGIN = "org.sonatype.plugins:nexus-staging-maven-plugin:1.7.0"


def identifier(value):
    # Treat service-issued IDs as opaque, not legacy hex profile IDs. Disallow
    # URL/control characters and Maven's comma-separated multi-repository form.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value or ""):
        raise ValueError("An explicit single staging profile/repository ID is required")
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Staging API redirect refused")


def request(path, payload=None):
    username = os.environ.get("CENTRAL_SONATYPE_TOKEN_USERNAME", "")
    password = os.environ.get("CENTRAL_SONATYPE_TOKEN_PASSWORD", "")
    if not username or not password:
        raise ValueError("Both Central Portal token secrets are required")
    auth = base64.b64encode(f"{username}:{password}".encode()).decode()
    req = Request(ENDPOINT + "/service/local/staging/" + path,
                  data=json.dumps(payload).encode() if payload is not None else None,
                  headers={"Authorization": "Basic " + auth, "Accept": "application/json",
                           "Content-Type": "application/json"})
    with build_opener(NoRedirect()).open(req, timeout=60) as response:
        return json.load(response)


def profiles():
    rows = request("profiles")["data"]
    if not isinstance(rows, list):
        raise ValueError("Invalid staging profile response")
    return rows


def check_open(profile, repository):
    identifier(profile)
    identifier(repository)
    # A profile-scoped query proves membership without assuming a manual API
    # repository key equals a Nexus stagingRepositoryId. Never select by IP.
    rows = request("profile_repositories/" + profile)["data"]
    matches = [row for row in rows if row.get("repositoryId") == repository]
    if len(matches) != 1:
        raise ValueError("Repository ID not uniquely present in the requested staging profile")
    row = matches[0]
    if row.get("type") != "open" or row.get("transitioning") is not False:
        raise ValueError("Staging repository must be OPEN and not transitioning")
    return row


def receipt(path, profile, repository):
    data = {"endpoint": ENDPOINT, "stagingProfileId": profile,
            "stagingRepositoryId": repository, "state": "open"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data), flush=True)
    return data


def open_repository(profile, output):
    identifier(profile)
    if not any(row.get("id") == profile for row in profiles()):
        raise ValueError("Profile ID not returned by the authenticated service")
    # Same Nexus start operation used by rc-open. Never retry POST: if the
    # response is lost, inspect the service rather than create another repo.
    data = request("profiles/" + profile + "/start",
                   {"data": {"description": "DL4J shared open staging"}})
    repository = identifier(data["data"]["stagedRepositoryId"])
    # Persist immediately, even if subsequent verification fails.
    receipt(output, profile, repository)
    check_open(profile, repository)
    return repository


def upload_command(repository, profile, repository_id, settings, maven="mvn"):
    identifier(profile)
    identifier(repository_id)
    return [*resolve_maven_command(maven), "--batch-mode", "--no-transfer-progress",
            "--settings", str(settings), PLUGIN + ":deploy-staged-repository",
            "-DnexusUrl=" + ENDPOINT + "/", "-DserverId=central",
            "-DstagingProfileId=" + profile, "-DstagingRepositoryId=" + repository_id,
            "-DrepositoryDirectory=" + str(repository.resolve()),
            "-DskipStagingRepositoryClose=true", "-DautoReleaseAfterClose=false",
            "-DkeepStagingRepositoryOnFailure=true"]


def upload(repository, profile, repository_id, output, maven="mvn"):
    check_open(profile, repository_id)
    sign_repository(repository)
    # Run outside the source reactor: no inherited extensions, lifecycle goals,
    # profiles or project .mvn configuration can turn this into a Portal upload.
    with tempfile.TemporaryDirectory(prefix="dl4j-open-staging-") as directory:
        settings = Path(directory) / "settings.xml"
        settings.write_text('''<settings><servers><server><id>central</id>
<username>${env.CENTRAL_SONATYPE_TOKEN_USERNAME}</username>
<password>${env.CENTRAL_SONATYPE_TOKEN_PASSWORD}</password>
</server></servers></settings>''', encoding="utf-8")
        check_open(profile, repository_id)
        subprocess.run(upload_command(repository, profile, repository_id, settings, maven),
                       cwd=directory, check=True)
    check_open(profile, repository_id)
    receipt(output, profile, repository_id)


def verify_java_parents(repository, log, version):
    """Compare ALL installed reactor POMs, including parents, before upload."""
    for relative, installed in inventory(log.read_text(), version).items():
        staged = repository / relative
        if not staged.is_file() or staged.read_bytes() != installed.read_bytes():
            raise ValueError(f"Missing or different Java reactor POM: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profiles", "open", "check", "upload"))
    parser.add_argument("--profile", default=os.environ.get("STAGING_PROFILE_ID", ""))
    parser.add_argument("--repository-id", default=os.environ.get("STAGING_REPOSITORY_ID", ""))
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--receipt", type=Path, default=Path("open-staging.json"))
    parser.add_argument("--java-log", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    if args.action == "profiles":
        print(json.dumps(profiles(), indent=2))
    elif args.action == "open":
        if args.repository_id:
            parser.error("open refuses an existing repository ID; use check or upload")
        open_repository(args.profile, args.receipt)
    elif args.action == "check":
        check_open(args.profile, args.repository_id)
    else:
        if args.repository is None:
            parser.error("upload requires --repository")
        if args.java_log:
            if not args.version:
                parser.error("--java-log requires --version")
            verify_java_parents(args.repository, args.java_log, args.version)
        upload(args.repository, args.profile, args.repository_id, args.receipt)


if __name__ == "__main__":
    main()
