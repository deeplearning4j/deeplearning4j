/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
 * direction <br>
 *  FWD: 0 = fwd
 *  BWD: 1 = bwd
 *  BIDIR_SUM: 2 = bidirectional sum
 *  BIDIR_CONCAT: 3 = bidirectional concat
 *  BIDIR_EXTRA_DIM: 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3) */
public enum LSTMDirectionMode {
  FWD,

  BWD,

  BIDIR_SUM,

  BIDIR_CONCAT,

  BIDIR_EXTRA_DIM
}
