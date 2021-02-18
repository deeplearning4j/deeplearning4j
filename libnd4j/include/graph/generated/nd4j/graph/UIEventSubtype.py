#  /* ******************************************************************************
#   *
#   *
#   * This program and the accompanying materials are made available under the
#   * terms of the Apache License, Version 2.0 which is available at
#   * https://www.apache.org/licenses/LICENSE-2.0.
#   *
#   *  See the NOTICE file distributed with this work for additional
#   *  information regarding copyright ownership.
#   * Unless required by applicable law or agreed to in writing, software
#   * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#   * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#   * License for the specific language governing permissions and limitations
#   * under the License.
#   *
#   * SPDX-License-Identifier: Apache-2.0
#   ******************************************************************************/

class UIEventSubtype(object):
    NONE = 0
    EVALUATION = 1
    LOSS = 2
    LEARNING_RATE = 3
    TUNING_METRIC = 4
    PERFORMANCE = 5
    PROFILING = 6
    FEATURE_LABEL = 7
    PREDICTION = 8
    USER_CUSTOM = 9

