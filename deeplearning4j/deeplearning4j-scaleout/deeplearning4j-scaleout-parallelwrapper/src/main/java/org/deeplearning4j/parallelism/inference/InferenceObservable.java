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

package org.deeplearning4j.parallelism.inference;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.Observer;

/**
 * @author raver119@gmail.com
 */
public interface InferenceObservable {

    /**
     * Get input batches - and their associated input mask arrays, if any<br>
     * Note that usually the returned list will be of size 1 - however, in the batched case, not all inputs
     * can actually be batched (variable size inputs to fully convolutional net, for example). In these "can't batch"
     * cases, multiple input batches will be returned, to be processed
     *
     * @return List of pairs of input arrays and input mask arrays. Input mask arrays may be null.
     */
    List<Pair<INDArray[],INDArray[]>> getInputBatches();

    void addInput(INDArray... input);

    void addInput(INDArray[] input, INDArray[] inputMasks);

    void setOutputBatches(List<INDArray[]> output);

    void setOutputException(Exception e);

    void addObserver(Observer observer);

    INDArray[] getOutput();
}
