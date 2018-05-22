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

if [[ "$2" != *-SNAPSHOT ]]
then
    echo "Error: Version $SNAPSHOT_VERSION should finish with -SNAPSHOT"
    exit 1
fi


echo "Releasing version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
echo "========================================================================================"

if [[ ! -z $(git tag -l "scalnet-$RELEASE_VERSION") ]]; then
    echo "Error: Version $RELEASE_VERSION has already been released!"
    exit 1
fi

sed -i "s/<nd4j.version>.*<\/nd4j.version>/<nd4j.version>$RELEASE_VERSION<\/nd4j.version>/" pom.xml
sed -i "s/<datavec.version>.*<\/datavec.version>/<datavec.version>$RELEASE_VERSION<\/datavec.version>/" pom.xml
sed -i "s/<dl4j.version>.*<\/dl4j.version>/<dl4j.version>$RELEASE_VERSION<\/dl4j.version>/" pom.xml
# In its normal state, repo should contain a snapshot version stanza
sed -i "s/<version>.*-SNAPSHOT<\/version>/<version>$RELEASE_VERSION<\/version>/" pom.xml
mvn versions:set -DallowSnapshots=false -DgenerateBackupPoms=false -DnewVersion=$RELEASE_VERSION

if [[ "${SKIP_BUILD}" == "0" ]]; then
    source change-scala-versions.sh 2.10
    mvn clean deploy -Dgpg.executable=gpg2 -DperformRelease -Psonatype-oss-release -DskipTests -DstagingRepositoryId=$STAGING_REPOSITORY -Dscalastyle.skip
    source change-scala-versions.sh 2.11
    mvn clean deploy -Dgpg.executable=gpg2 -DperformRelease -Psonatype-oss-release -DskipTests -DstagingRepositoryId=$STAGING_REPOSITORY -Dscalastyle.skip

    source change-scala-versions.sh 2.11
fi
git commit -s -a -m "Update to version $RELEASE_VERSION"
git tag -s -a -m "scalnet-$RELEASE_VERSION" "scalnet-$RELEASE_VERSION"
git tag -s -a -f -m "scalnet-$RELEASE_VERSION" "latest_release"

sed -i "s/<nd4j.version>.*<\/nd4j.version>/<nd4j.version>$SNAPSHOT_VERSION<\/nd4j.version>/" pom.xml
sed -i "s/<datavec.version>.*<\/datavec.version>/<datavec.version>$SNAPSHOT_VERSION<\/datavec.version>/" pom.xml
sed -i "s/<dl4j.version>.*<\/dl4j.version>/<dl4j.version>$SNAPSHOT_VERSION<\/dl4j.version>/" pom.xml
# back to a version stanza
sed -i "s/<version>$RELEASE_VERSION<\/version>/<version>$SNAPSHOT_VERSION<\/version>/" pom.xml
mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion=$SNAPSHOT_VERSION
git commit -s -a -m "Update to version $SNAPSHOT_VERSION"

echo "Successfully performed release of version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $STAGING_REPOSITORY"
