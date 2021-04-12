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

set -eu

if [[ $# < 2 ]]; then
    echo "Usage: bash release-specified-component.sh release_version snapshot_version [staging_repository]"
    exit 1
fi

RELEASE_VERSION=$1
SNAPSHOT_VERSION=$2
#STAGING_REPOSITORY=
STAGING_REPOSITORY=${3:-}
DEPLOY_COMMAND=$4
STAGING_REPO_FLAG=
RELEASE_PROFILE=${RELEASE_PROFILE:-sonatype}
if [[ "$2" != *-SNAPSHOT ]]
then
    echo "Error: Version $SNAPSHOT_VERSION should finish with -SNAPSHOT"
    exit 1
fi

echo "Releasing version $RELEASE_VERSION ($SNAPSHOT_VERSION) to repository $RELEASE_PROFILE $STAGING_REPOSITORY"
echo "========================================================================================================="

if [[ ! -z $(git tag -l "deeplearning4j-$RELEASE_VERSION") ]]; then
    echo "Error: Version $RELEASE_VERSION has already been released!"
    exit 1
fi

bash ./update-versions.sh "$SNAPSHOT_VERSION" "$RELEASE_VERSION"

#    mvn clean deploy -Dgpg.executable=gpg2 -Prelease -Dlocal.software.repository=$RELEASE_PROFILE -Dmaven.test.skip -Dlibnd4j.cuda=8.0 -Denforcer.skip -DstagingRepositoryId=$STAGING_REPOSITORY
# add flags specific to release, assumes an mvn deploy command is given
# Note: we add javadoc fail on error here to avoid javadoc errors blocking a release
DEPLOY_COMMAND="${DEPLOY_COMMAND}  -Prelease  -Dlocal.software.repository=$RELEASE_PROFILE -Denforcer.skip -Dmaven.javadoc.failOnError=false -Ddl4j.release.server=ossrh"
 #-DstagingRepositoryId=$STAGING_REPOSITORY
if [[ -z ${STAGING_REPOSITORY:-} ]]; then
    # create new staging repository with everything in it
    if [[ ! $(grep stagingRepository.id target/nexus-staging/staging/*.properties) =~ ^stagingRepository.id=(.*) ]]; then
        echo "Setting empty staging repository"
        STAGING_REPOSITORY=
        STAGING_REPO_FLAG=
    else
        echo "Using staging repository ${STAGING_REPOSITORY}"
        #STAGING_REPOSITORY="${BASH_REMATCH[1]}"
        #only specify a staging repo if it's not empty
        STAGING_REPO_FLAG="-DstagingRepositoryId=${STAGING_REPOSITORY}"
    fi
fi

DEPLOY_COMMAND="${DEPLOY_COMMAND} ${STAGING_REPO_FLAG}"
echo "RUNNING maven command ${DEPLOY_COMMAND}"
eval "$DEPLOY_COMMAND"

