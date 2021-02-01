#!/bin/bash
#
# /* ******************************************************************************
#  * Copyright (c) 2021 Deeplearning4j Contributors
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#

# Note: this needs manual upgrading of version in setup.py to work (can't override old versions)

# remove old wheels
sudo rm -rf dist/*

# Build Python 2 & 3 wheels for current version
sudo python2 setup.py sdist bdist_wheel
sudo python3 setup.py sdist bdist_wheel

# Upload to PyPI with twine. Needs full "skymind" credentials in ~/.pypirc
twine upload dist/*