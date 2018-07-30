#!/usr/bin/env bash

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

shopt -s dotglob # to be able to mv all hidden files easily

mkdir deeplearning4j
git mv -k * deeplearning4j
git commit -a -m "Move deeplearning4j"

PROJECTS=(libnd4j nd4j datavec arbiter nd4s gym-java-client rl4j scalnet jumpy)
for PROJECT in ${PROJECTS[@]}; do
    git branch -D $PROJECT || true
    git checkout --orphan $PROJECT
    git reset --hard
    rm -Rf $PROJECT
    git pull https://github.com/deeplearning4j/$PROJECT master
    mkdir $PROJECT
    git mv -k * $PROJECT
    git commit -a -m "Move $PROJECT"
    git checkout feature/monorepo
    git merge --allow-unrelated-histories $PROJECT -m "Merge $PROJECT"
done

