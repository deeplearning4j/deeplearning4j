#!/bin/bash
set -eu

if [[ $# < 2 ]]; then
    echo "Usage: bash perform-release.sh release_version snapshot_version [staging_repository]"
    exit 1
fi

RELEASE_VERSION=$1
SNAPSHOT_VERSION=$2
STAGING_REPOSITORY=${3:-}

echo "Releasing version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
echo "========================================================================================"

if [[ ! -z $(git tag -l "nd4j-$RELEASE_VERSION") ]]; then
    echo "Error: Version $RELEASE_VERSION has already been released!"
    exit 1
fi

cd ../libnd4j
export TRICK_NVCC=YES
export LIBND4J_HOME="$(pwd)"
if [[ -z $(git tag -l "libnd4j-$RELEASE_VERSION") ]]; then
    bash buildnativeoperations.sh -c cpu
    bash buildnativeoperations.sh -c cuda -v 7.5
    bash buildnativeoperations.sh -c cuda -v 8.0
    git tag -a -m "libnd4j-$RELEASE_VERSION" "libnd4j-$RELEASE_VERSION"
fi
cd ../nd4j

mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion=$RELEASE_VERSION

if [[ -z ${STAGING_REPOSITORY:-} ]]; then
    # create new staging repository with everything in it
    source change-scala-versions.sh 2.10
    source change-cuda-versions.sh 7.5
    mvn clean deploy -Dgpg.executable=gpg2 -DperformRelease -Psonatype-oss-release -DskipTests -DstagingRepositoryId=$STAGING_REPOSITORY

    if [[ ! $(grep stagingRepository.id target/nexus-staging/staging/*.properties) =~ ^stagingRepository.id=(.*) ]]; then
        exit 1
    fi
    STAGING_REPOSITORY=${BASH_REMATCH[1]}

    source change-scala-versions.sh 2.11
    source change-cuda-versions.sh 8.0
    mvn clean deploy -Dgpg.executable=gpg2 -DperformRelease -Psonatype-oss-release -DskipTests -DstagingRepositoryId=$STAGING_REPOSITORY
else
    # build only partially (on other platforms) and deploy to given repository
    source change-scala-versions.sh 2.10
    source change-cuda-versions.sh 7.5
    mvn clean deploy -Dgpg.useagent=false -DperformRelease -Psonatype-oss-release -DskipTests -Dmaven.javadoc.skip -DstagingRepositoryId=$STAGING_REPOSITORY

    source change-scala-versions.sh 2.11
    source change-cuda-versions.sh 8.0
    mvn clean deploy -Dgpg.useagent=false -DperformRelease -Psonatype-oss-release -DskipTests -Dmaven.javadoc.skip -DstagingRepositoryId=$STAGING_REPOSITORY
fi

source change-scala-versions.sh 2.10
source change-cuda-versions.sh 7.5
git commit -a -m "Update to version $RELEASE_VERSION"
git tag -a -m "nd4j-$RELEASE_VERSION" "nd4j-$RELEASE_VERSION"

mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion=$SNAPSHOT_VERSION
git commit -a -m "Update to version $SNAPSHOT_VERSION"

echo "Successfully performed release of version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
