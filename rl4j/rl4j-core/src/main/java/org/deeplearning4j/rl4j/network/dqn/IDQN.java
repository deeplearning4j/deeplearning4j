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

package org.deeplearning4j.rl4j.network.dqn;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 *
 * This neural net quantify the value of each action given a state
 *
 */
public interface IDQN<NN extends IDQN> extends NeuralNet<NN> {

    boolean isRecurrent();

    void reset();

    void fit(INDArray input, INDArray labels);

    void fit(INDArray input, INDArray[] labels);

    INDArray output(INDArray batch);

    INDArray[] outputAll(INDArray batch);

    NN clone();

    void copy(NN from);

    Gradient[] gradient(INDArray input, INDArray label);

    Gradient[] gradient(INDArray input, INDArray[] label);

    void applyGradient(Gradient[] gradient, int batchSize);

    double getLatestScore();
}
