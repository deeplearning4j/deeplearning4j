#!/bin/bash
set -eu

if [[ $# < 2 ]]; then
    echo "Usage: bash perform-release.sh release_version snapshot_version staging_repository"
    exit 1
fi

RELEASE_VERSION=$1
SNAPSHOT_VERSION=$2
STAGING_REPOSITORY=${3:-}
SKIP_BUILD=${SKIP_BUILD:-0}
RELEASE_PROFILE=${RELEASE_PROFILE:-sonatype}

echo "Releasing version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
echo "========================================================================================"

if [[ ! -z $(git tag -l "nd4s-$RELEASE_VERSION") ]]; then
    echo "Error: Version $RELEASE_VERSION has already been released!"
    exit 1
fi

sed -i "s/\"currentVersion\", default = \".*\"/\"currentVersion\", default = \"$RELEASE_VERSION\"/" build.sbt
sed -i "s/\"nd4jVersion\", default = \".*\"/\"nd4jVersion\", default = \"$RELEASE_VERSION\"/" build.sbt

# ~/.ivy2/.credentials needs to look like this:
# realm=Sonatype Nexus Repository Manager
# host=oss.sonatype.org
# user=xxx
# password=xxx
if [[ "${SKIP_BUILD}" == "0" ]]; then
    sbt -DrepoType=$RELEASE_PROFILE -DstageRepoId=$STAGING_REPOSITORY +publishSigned
fi

git commit -s -a -m "Update to version $RELEASE_VERSION"
git tag -s -a -m "nd4s-$RELEASE_VERSION" "nd4s-$RELEASE_VERSION"
git tag -s -a -f -m "nd4s-$RELEASE_VERSION" "latest_release"

sed -i "s/\"currentVersion\", default = \".*\"/\"currentVersion\", default = \"$SNAPSHOT_VERSION\"/" build.sbt
sed -i "s/\"nd4jVersion\", default = \".*\"/\"nd4jVersion\", default = \"$SNAPSHOT_VERSION\"/" build.sbt
git commit -s -a -m "Update to version $SNAPSHOT_VERSION"

echo "Successfully performed release of version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
