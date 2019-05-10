/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.rl4j.learning.async;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Its called a MiniTrans because it is similar to a Transition
 * but without a next observation
 *
 * It is stacked and then processed by AsyncNStepQL or A3C
 * following the paper implementation https://arxiv.org/abs/1602.01783 paper.
 *
 */
@AllArgsConstructor
@Value
public class MiniTrans<A> {
    INDArray obs;
    A action;
    INDArray[] output;
    double reward;
}
