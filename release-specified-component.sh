#!/bin/bash

#
# /* ******************************************************************************
#  *
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  *  See the NOTICE file distributed with this work for additional
#  *  information regarding copyright ownership.
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#


if [[ $# < 2 ]]; then
    echo "Usage: bash release-specified-component.sh release_version snapshot_version [staging_repository]"
    exit 1
fi


RELEASE_VERSION=$1
SNAPSHOT_VERSION=$2
#STAGING_REPOSITORY=
STAGING_REPOSITORY=$3

if [ -z "$COMMAND" ]; then
  if [ -z "$4" ]; then
    echo "Please specify a command environment variable or a 4th parameter specifying the maven command."
    exit 1
    else
        COMMAND="$4"
  fi

fi

DEPLOY_COMMAND="$COMMAND"
STAGING_REPO_FLAG=
RELEASE_PROFILE=${RELEASE_PROFILE:-sonatype}
if [[ "$2" != *-SNAPSHOT ]]
then
    echo "Error: Version $SNAPSHOT_VERSION should finish with -SNAPSHOT"
    exit 1
fi

echo "Releasing version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $RELEASE_PROFILE $STAGING_REPOSITORY"
echo "========================================================================================================="

#if [[ ! -z $(git tag -l "deeplearning4j-$RELEASE_VERSION") ]]; then
#    echo "Error: Version $RELEASE_VERSION has already been released!"
#    exit 1
#fi

bash ./update-versions.sh "$SNAPSHOT_VERSION" "$RELEASE_VERSION"

#    mvn clean deploy -Dgpg.executable=gpg2 -Prelease -Dlocal.software.repository=$RELEASE_PROFILE -Dmaven.test.skip -Dlibnd4j.cuda=8.0 -Denforcer.skip -DstagingRepositoryId=$STAGING_REPOSITORY
# add flags specific to release, assumes an mvn deploy command is given
# Note: we add javadoc fail on error here to avoid javadoc errors blocking a release
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo ${machine}

if [[ "${machine}" == "Mac" ]]; then
   echo "Adding osx specific parameters for gpg signing"
   #DEPLOY_COMMAND="${DEPLOY_COMMAND}  -X  "
fi

if [ -z "${MODULES}" ]; then export MODULES= ; fi


DEPLOY_COMMAND="${DEPLOY_COMMAND}  ${MODULES} -Prelease  -Dlocal.software.repository=$RELEASE_PROFILE -Denforcer.skip -Dmaven.javadoc.failOnError=false -Ddl4j.release.server=ossrh"

 #-DstagingRepositoryId=$STAGING_REPOSITORY
if [[ -z ${STAGING_REPOSITORY:-} ]]; then
    # create new staging repository with everything in it
    if [[ ! $(grep stagingRepository.id target/nexus-staging/staging/*.properties) =~ ^stagingRepository.id=(.*) ]]; then
        echo "Setting empty staging repository"
        STAGING_REPOSITORY=
        # note: the reason we set this is to ensure when bootstrapping a build we're not trying to download from a repo that doesn't exist
        # this --also-make flag handles any transitive dependencies and uploads those. The reason this is important is if a build tries to upload
        # an already existing dependency, sonatype nexus returns a 400 bad request. So we need deps that haven't already been uploaded.
        # this is because repos are immutable by default, no overrides are allowed whereas for snapshots this is not the case.
        STAGING_REPO_FLAG=" --also-make"
    else
        echo "Using staging repository ${STAGING_REPOSITORY}"
        #STAGING_REPOSITORY="${BASH_REMATCH[1]}"
        #only specify a staging repo if it's not empty
        STAGING_REPO_FLAG="-DstagingRepositoryId=${STAGING_REPOSITORY} --also-make"
    fi
    else
        STAGING_REPO_FLAG="-DstagingRepositoryId=${STAGING_REPOSITORY} --also-make"
        echo "USING STAGING_REPOSITORY ${STAGING_REPOSITORY}, setting flag to ${STAGING_REPO_FLAG}"
fi

DEPLOY_COMMAND="${DEPLOY_COMMAND} ${STAGING_REPO_FLAG}"
echo "RUNNING maven command ${DEPLOY_COMMAND}"
eval "$DEPLOY_COMMAND"

