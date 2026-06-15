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

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.enums;

/**
 * Activations
 */
public enum GateAct {
  TANH(0),

  RELU(1),

  SIGMOID(2),

  AFFINE(3),

  LEAKY_RELU(4),

  THRESHHOLD_RELU(5),

  SCALED_TAHN(6),

  HARD_SIGMOID(7),

  ELU(8),

  SOFTSIGN(9),

  SOFTPLUS(10);

  private final int methodIndex;

  GateAct(int index) {
    this.methodIndex = index;
  }

  public int methodIndex() {
    return methodIndex;
  }
}
