#!/bin/bash -u
#
# push_to_bintray.sh - francois@skymind.io
#
# This script push a package to Bintray repo
#

function usage() {
  echo "$0 username api_key organisation package_file site_url"
  exit 0
}

if [ $# -lt 5 ]; then
 usage
fi

BINTRAY_USER=$1
BINTRAY_APIKEY=$2
BINTRAY_ACCOUNT=$3
PACKAGE_FILE=$4
BASE_DESC=$5
FILE_EXTENSION=${PACKAGE_FILE##*.}
BINTRAY_REPO="${FILE_EXTENSION,,}"
XDEBUG=""

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

CURL_SILENT_CMD="curl --write-out %{http_code} --location --silent --output /dev/null -u$BINTRAY_USER:$BINTRAY_APIKEY"
CURL_VERBOSE_CMD="curl --write-out %{http_code} --location -u$BINTRAY_USER:$BINTRAY_APIKEY"

CURL_CMD=$CURL_SILENT_CMD

function grab() {
    local REGEX="s/set\($1 \\\"([^\\\"]*)\\\"\)$/\1/p"
    local RES=$(sed -nEe "$REGEX" $DIR/../CMakeLists.txt)
    echo $RES
}

PACKAGE_NAME=$(grab "CPACK_PACKAGE_NAME")
PACKAGE_DESCRIPTION=$(grab "CPACK_PACKAGE_DESCRIPTION_SUMMARY")
PACKAGE_LICENSE=$(grab "CPACK_RPM_PACKAGE_LICENSE")
PACKAGE_RELEASE=$(basename $(dirname $PACKAGE_FILE))
PACKAGE_ARCH=$(uname -i)
PACKAGE_MAJOR_VERSION=$(grab "CPACK_PACKAGE_VERSION_MAJOR")
PACKAGE_MINOR_VERSION=$(grab "CPACK_PACKAGE_VERSION_MINOR")
PACKAGE_PATCH_VERSION=$(grab "CPACK_PACKAGE_VERSION_PATCH")
PACKAGE_VERSION="$PACKAGE_MAJOR_VERSION.$PACKAGE_MINOR_VERSION.$PACKAGE_PATCH_VERSION"

REPO_FILE_PATH="$PACKAGE_RELEASE/$(basename $PACKAGE_FILE)"
DESC_URL=$BASE_DESC/$PACKAGE_NAME

if [ -z "$PACKAGE_NAME" ] || [ -z "$PACKAGE_VERSION" ] || [ -z "$PACKAGE_RELEASE" ] || [ -z "$PACKAGE_ARCH" ]; then
  echo "no PACKAGE metadata information in $PACKAGE_FILE, skipping."
  exit -1
fi

if [ $BINTRAY_REPO == "deb" ]; then
    DEB_ARCH_ARG="X-Bintray-Debian-Architecture:$PACKAGE_ARCH"
    DEB_COMPONENT_ARG="X-Bintray-Debian-Component:main"
    DEB_DISTRIBUTION_ARG="X-Bintray-Debian-Distribution:$(lsb_release -sc)"
    DEB_ARGS="-H $DEB_ARCH_ARG -H $DEB_COMPONENT_ARG -H $DEB_DISTRIBUTION_ARG"
else
    DEB_ARGS=""
fi

echo "PACKAGE_NAME=$PACKAGE_NAME, PACKAGE_VERSION=$PACKAGE_VERSION, PACKAGE_RELEASE=$PACKAGE_RELEASE, PACKAGE_ARCH=$PACKAGE_ARCH"
echo "BINTRAY_USER=$BINTRAY_USER, BINTRAY_ACCOUNT=$BINTRAY_ACCOUNT, BINTRAY_REPO=$BINTRAY_REPO, PACKAGE_FILE=$PACKAGE_FILE, BASE_DESC=$BASE_DESC"

echo "Deleting version from Bintray.."
HTTP_CODE=`$CURL_CMD -H "Content-Type: application/json" -X DELETE https://api.bintray.com/packages/$BINTRAY_ACCOUNT/$BINTRAY_REPO/$PACKAGE_NAME/versions/$PACKAGE_VERSION-$PACKAGE_RELEASE`

if [ "$HTTP_CODE" != "200" ]; then
 echo "can't delete package -> $HTTP_CODE"
else
 echo "Package deleted"
fi

echo "Creating package on Bintray.."
DATA_JSON="{ \"name\": \"$PACKAGE_NAME\", \"desc\": \"${PACKAGE_DESCRIPTION}\", \"vcs_url\": \"$DESC_URL\", \"labels\": \"\", \"licenses\": [ \"$PACKAGE_LICENSE\" ] }"

if [ "$XDEBUG" = "true" ]; then
 echo "DATA_JSON=$DATA_JSON"
fi

HTTP_CODE=`$CURL_CMD -H "Content-Type: application/json" -X POST https://api.bintray.com/packages/$BINTRAY_ACCOUNT/$BINTRAY_REPO/ --data "$DATA_JSON"`

if [ "$HTTP_CODE" != "201" ]; then
    echo "can't create package -> $HTTP_CODE"
    echo "Assuming package already exists"
else
    echo "Package created"
fi

echo "Uploading package to Bintray.."
HTTP_CODE=`$CURL_CMD -T $PACKAGE_FILE -u$BINTRAY_USER:$BINTRAY_APIKEY -H "X-Bintray-Package:$PACKAGE_NAME" -H "X-Bintray-Version:$PACKAGE_VERSION-$PACKAGE_RELEASE" $DEB_ARGS "https://api.bintray.com/content/$BINTRAY_ACCOUNT/$BINTRAY_REPO/$REPO_FILE_PATH;publish=1"`

if [ "$HTTP_CODE" != "201" ]; then
 echo "failed to upload package -> $HTTP_CODE"
 exit -1
else
 echo "Package uploaded"
fi

exit 0
