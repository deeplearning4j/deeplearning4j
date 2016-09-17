#!/bin/bash
set -eu

if [[ $# < 2 ]]; then
    echo "Usage: bash perform-release.sh release_version snapshot_version staging_repository"
    exit 1
fi

RELEASE_VERSION=$1
SNAPSHOT_VERSION=$2
STAGING_REPOSITORY=${3:-}

echo "Releasing version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
echo "========================================================================================"

if [[ ! -z $(git tag -l "nd4s-$RELEASE_VERSION") ]]; then
    echo "Error: Version $RELEASE_VERSION has already been released!"
    exit 1
fi

sed -i "s/version := \".*\",/version := \"$RELEASE_VERSION\",/" build.sbt
sed -i "s/nd4jVersion := \".*\",/nd4jVersion := \"$RELEASE_VERSION\",/" build.sbt

# ~/.ivy2/.credentials needs to look like this:
# realm=Sonatype Nexus Repository Manager
# host=oss.sonatype.org
# user=xxx
# password=xxx
sbt +publishSigned

git commit -a -m "Update to version $RELEASE_VERSION"
git tag -a -m "nd4s-$RELEASE_VERSION" "nd4s-$RELEASE_VERSION"

sed -i "s/version := \".*\",/version := \"$SNAPSHOT_VERSION\",/" build.sbt
sed -i "s/nd4jVersion := \".*\",/nd4jVersion := \"$SNAPSHOT_VERSION\",/" build.sbt
git commit -a -m "Update to version $SNAPSHOT_VERSION"

echo "Successfully performed release of version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
