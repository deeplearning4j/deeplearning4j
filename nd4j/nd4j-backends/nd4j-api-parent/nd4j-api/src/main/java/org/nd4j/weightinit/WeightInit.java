/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.weightinit;

public enum WeightInit {
    DISTRIBUTION, ZERO, ONES, SIGMOID_UNIFORM, NORMAL, LECUN_NORMAL, UNIFORM, XAVIER, XAVIER_UNIFORM, XAVIER_FAN_IN, XAVIER_LEGACY, RELU,
    RELU_UNIFORM, IDENTITY, LECUN_UNIFORM, VAR_SCALING_NORMAL_FAN_IN, VAR_SCALING_NORMAL_FAN_OUT, VAR_SCALING_NORMAL_FAN_AVG,
    VAR_SCALING_UNIFORM_FAN_IN, VAR_SCALING_UNIFORM_FAN_OUT, VAR_SCALING_UNIFORM_FAN_AVG,SUPPLIED
}
