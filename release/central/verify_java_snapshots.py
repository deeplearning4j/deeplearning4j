"""Verify every Java reactor POM, including parents, against public snapshots.

The Maven install log is the inventory, not central-deferred: skipped publishing
modules are absent from central-deferred but still required by consumers.
"""
import argparse
from pathlib import Path
import re
import time
from urllib.request import urlopen
import xml.etree.ElementTree as ET


SNAPSHOTS = "https://central.sonatype.com/repository/maven-snapshots"
REQUIRED_PARENTS = {"nd4j-backends", "nd4j-api-parent", "nd4j-backend-impls"}
INSTALL = re.compile(
    r"Installing .+ to (.+/repository/(org/eclipse/deeplearning4j/[^/]+/[^/]+/[^/]+\.pom))$")


def inventory(log, version):
    poms = {}
    for line in log.splitlines():
        match = INSTALL.search(line)
        if match:
            local, relative = match.groups()
            if relative.split("/")[-2] == version:
                poms[relative] = Path(local)
    artifacts = {path.split("/")[-3] for path in poms}
    missing = REQUIRED_PARENTS - artifacts
    if missing:
        raise ValueError(f"Missing required reactor parent POMs: {sorted(missing)}")
    if "Skipping Central Snapshot Publishing" in log:
        raise ValueError("Maven skipped snapshot publication for a reactor module")
    return poms


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def verify(poms, attempts=5, delay=15):
    for relative, local in sorted(poms.items()):
        directory = relative.rsplit("/", 1)[0]
        artifact, version = directory.split("/")[-2:]
        expected = local.read_bytes()
        for attempt in range(attempts):
            try:
                metadata = ET.fromstring(fetch(f"{SNAPSHOTS}/{directory}/maven-metadata.xml"))
                values = [entry.findtext("value") for entry in metadata.findall(
                    "./versioning/snapshotVersions/snapshotVersion")
                    if entry.findtext("extension") == "pom" and not entry.findtext("classifier")]
                if len(values) != 1 or not values[0] or "/" in values[0]:
                    raise ValueError(f"Missing or ambiguous POM snapshot metadata: {relative}")
                remote = fetch(f"{SNAPSHOTS}/{directory}/{artifact}-{values[0]}.pom")
                if remote != expected:
                    raise ValueError(f"Published POM differs from this build: {relative}")
                break
            except (OSError, ValueError, ET.ParseError):
                if attempt + 1 == attempts:
                    raise
                time.sleep(delay)
        print(f"VERIFIED {artifact}:{version} -> {values[0]} (POM bytes match)", flush=True)
    print(f"Verified {len(poms)} published reactor POMs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.version.endswith("-SNAPSHOT"):
        parser.error("--version must end with -SNAPSHOT")
    verify(inventory(args.log.read_text(), args.version))


if __name__ == "__main__":
    main()
