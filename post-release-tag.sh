#!/bin/bash

git commit -s -a -m "Update to version $RELEASE_VERSION"
git tag -s -a -m "deeplearning4j-$RELEASE_VERSION" "deeplearning4j-$RELEASE_VERSION"
git tag -s -a -f -m "deeplearning4j-$RELEASE_VERSION" "latest_release"

bash ./update-versions.sh "$RELEASE_VERSION" "$SNAPSHOT_VERSION"

git commit -s -a -m "Update to version $SNAPSHOT_VERSION"

echo "Successfully performed release of version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $RELEASE_PROFILE $STAGING_REPOSITORY"
