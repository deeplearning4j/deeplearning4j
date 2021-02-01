#!/bin/sh -x
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

TESTFILE=Makefile

(./zcgzip < ${TESTFILE} | gunzip | cmp - ${TESTFILE}) && \
(gzip < ${TESTFILE} | ./zcgunzip | cmp - ${TESTFILE})

# Result of "(cmd) && (cmd)" implicitly becomes result of this script
# and thus the test.
