#!/bin/bash
set -eu

if [[ $# < 2 ]]; then
    echo "Usage: bash perform-release.sh release_version snapshot_version [staging_repository]"
    exit 1
fi

RELEASE_VERSION=$1
SNAPSHOT_VERSION=$2
STAGING_REPOSITORY=${3:-}
SKIP_BUILD=${SKIP_BUILD:-0}
RELEASE_PROFILE=${RELEASE_PROFILE:-sonatype}

echo "Releasing version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $RELEASE_PROFILE $STAGING_REPOSITORY"
echo "========================================================================================================="

if [[ ! -z $(git tag -l "libnd4j-$RELEASE_VERSION") ]]; then
    echo "Error: Version $RELEASE_VERSION has already been released!"
    exit 1
fi

mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion=$RELEASE_VERSION

MAVEN_OPTIONS="-Dgpg.executable=gpg2 -DperformRelease -Dlocal.software.repository=$RELEASE_PROFILE -Dmaven.test.skip -Denforcer.skip -DstagingRepositoryId=$STAGING_REPOSITORY"

if [[ "${SKIP_BUILD}" == "0" ]]; then
    export TRICK_NVCC=YES
    export LIBND4J_HOME="$(pwd)"
    mvn clean deploy $MAVEN_OPTIONS
    mvn clean deploy $MAVEN_OPTIONS -Dlibnd4j.cuda=8.0
    mvn clean deploy $MAVEN_OPTIONS -Dlibnd4j.cuda=9.0
    mvn clean deploy $MAVEN_OPTIONS -Dlibnd4j.cuda=9.1
    mvn clean deploy $MAVEN_OPTIONS -Dlibnd4j.platform=android-arm
    mvn clean deploy $MAVEN_OPTIONS -Dlibnd4j.platform=android-arm64
    mvn clean deploy $MAVEN_OPTIONS -Dlibnd4j.platform=android-x86
    mvn clean deploy $MAVEN_OPTIONS -Dlibnd4j.platform=android-x86_64
fi

git commit -s -a -m "Update to version $RELEASE_VERSION"
git tag -s -a -m "libnd4j-$RELEASE_VERSION" "libnd4j-$RELEASE_VERSION"
git tag -s -a -f -m "libnd4j-$RELEASE_VERSION" "latest_release"

mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion=$SNAPSHOT_VERSION
git commit -s -a -m "Update to version $SNAPSHOT_VERSION"

echo "Successfully performed release of version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $RELEASE_PROFILE $STAGING_REPOSITORY"
