#!/bin/bash
OLD_VERSION=$1
VERSION=$2
sed -i "s/<nd4j.version>.*<\/nd4j.version>/<nd4j.version>$VERSION<\/nd4j.version>/" pom.xml
sed -i "s/<datavec.version>.*<\/datavec.version>/<datavec.version>$VERSION<\/datavec.version>/" pom.xml
sed -i "s/<dl4j.version>.*<\/dl4j.version>/<dl4j.version>$VERSION<\/dl4j.version>/" pom.xml
sed -i "s/<deeplearning4j.version>.*<\/deeplearning4j.version>/<deeplearning4j.version>$VERSION<\/deeplearning4j.version>/" pom.xml
sed -i "s/<dl4j-test-resources.version>.*<\/dl4j-test-resources.version>/<dl4j-test-resources.version>$VERSION<\/dl4j-test-resources.version>/" pom.xml
#Spark versions, like <version>xxx_spark_2-SNAPSHOT</version>
for f in $(find . -name 'pom.xml' -not -path '*target*'); do
    sed -i "s/version>.*_spark_.*</version>${VERSION}_spark_2</g" $f
done
mvn versions:set -DallowSnapshots=true -DgenerateBackupPoms=false -DnewVersion="$VERSION"
# back to a version stanza
sed -i "s/<version>$OLD_VERSION<\/version>/<version>$VERSION<\/version>/" pom.xml
