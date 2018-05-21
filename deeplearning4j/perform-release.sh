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

if [[ ! -z $(git tag -l "deeplearning4j-$RELEASE_VERSION") ]]; then
    echo "Error: Version $RELEASE_VERSION has already been released!"
    exit 1
fi

sed -i "s/<nd4j.version>.*<\/nd4j.version>/<nd4j.version>$RELEASE_VERSION<\/nd4j.version>/" pom.xml
sed -i "s/<datavec.version>.*<\/datavec.version>/<datavec.version>$RELEASE_VERSION<\/datavec.version>/" pom.xml
sed -i "s/<deeplearning4j.version>.*<\/deeplearning4j.version>/<deeplearning4j.version>$RELEASE_VERSION<\/deeplearning4j.version>/" pom.xml
sed -i "s/<dl4j-test-resources.version>.*<\/dl4j-test-resources.version>/<dl4j-test-resources.version>$RELEASE_VERSION<\/dl4j-test-resources.version>/" pom.xml
#Spark versions, like <version>xxx_spark_2-SNAPSHOT</version>
for f in $(find . -name 'pom.xml' -not -path '*target*'); do
    sed -i "s/version>.*_spark_.*</version>${RELEASE_VERSION}_spark_1</g" $f
done
mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion=$RELEASE_VERSION

if [[ "${SKIP_BUILD}" == "0" ]]; then
    source change-spark-versions.sh 1
    source change-scala-versions.sh 2.10
    source change-cuda-versions.sh 8.0
    mvn clean deploy -Dgpg.executable=gpg2 -DperformRelease -Dlocal.software.repository=$RELEASE_PROFILE -Dmaven.test.skip -DstagingRepositoryId=$STAGING_REPOSITORY
    source change-scala-versions.sh 2.11
    source change-cuda-versions.sh 9.0
    mvn clean deploy -Dgpg.executable=gpg2 -DperformRelease -Dlocal.software.repository=$RELEASE_PROFILE -Dmaven.test.skip -DstagingRepositoryId=$STAGING_REPOSITORY
    source change-spark-versions.sh 2
    mvn clean deploy -Dgpg.executable=gpg2 -DperformRelease -Dlocal.software.repository=$RELEASE_PROFILE -Dmaven.test.skip -DstagingRepositoryId=$STAGING_REPOSITORY -Dspark.major.version=2

    source change-spark-versions.sh 1
    source change-scala-versions.sh 2.11
    source change-cuda-versions.sh 9.0
fi
git commit -s -a -m "Update to version $RELEASE_VERSION"
git tag -s -a -m "deeplearning4j-$RELEASE_VERSION" "deeplearning4j-$RELEASE_VERSION"
git tag -s -a -f -m "deeplearning4j-$RELEASE_VERSION" "latest_release"

sed -i "s/<nd4j.version>.*<\/nd4j.version>/<nd4j.version>$SNAPSHOT_VERSION<\/nd4j.version>/" pom.xml
sed -i "s/<datavec.version>.*<\/datavec.version>/<datavec.version>$SNAPSHOT_VERSION<\/datavec.version>/" pom.xml
sed -i "s/<deeplearning4j.version>.*<\/deeplearning4j.version>/<deeplearning4j.version>$SNAPSHOT_VERSION<\/deeplearning4j.version>/" pom.xml
sed -i "s/<dl4j-test-resources.version>.*<\/dl4j-test-resources.version>/<dl4j-test-resources.version>$SNAPSHOT_VERSION<\/dl4j-test-resources.version>/" pom.xml
#Spark versions, like <version>xxx_spark_2-SNAPSHOT</version>
SPLIT_VERSION=(${SNAPSHOT_VERSION//-/ })
for f in $(find . -name 'pom.xml' -not -path '*target*'); do
    sed -i "s/version>.*_spark_.*</version>${SPLIT_VERSION[0]}_spark_1-${SPLIT_VERSION[1]}</g" $f
done
mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion=$SNAPSHOT_VERSION
git commit -s -a -m "Update to version $SNAPSHOT_VERSION"

echo "Successfully performed release of version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $RELEASE_PROFILE $STAGING_REPOSITORY"
