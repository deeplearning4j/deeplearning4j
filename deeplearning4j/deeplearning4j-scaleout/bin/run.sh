#!/bin/sh
################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

 java -cp "lib/*"  -Dhazelcast.access-key="$AWSAccessKeyId" -Dhazelcast.access-secret="AWSSecretKey"  -Dhazelcast.region="us-east-1d" -Dhazelcast.aws=true -Dhazelcast.host="masterhost"  -Xmx5g -Xms5g -server -XX:+UseTLAB   -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:MaxTenuringThreshold=0 -XX:CMSInitiatingOccupancyFraction=60  -XX:+CMSParallelRemarkEnabled -XX:+CMSPermGenSweepingEnabled -XX:+CMSClassUnloadingEnabled org.deeplearning4j.org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunnerApp -h ec2-54-86-216-254.compute-1.amazonaws.com -t worker
