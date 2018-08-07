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

package org.deeplearning4j.optimize.api;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.List;
import java.util.Map;

/**
 * A listener interface for training DL4J models.<br>
 * The methods here will be called at various points during training, and only during training.<br>
 * Note that users can extend {@link BaseTrainingListener} and selectively override the required methods,
 * instead of implementing TrainingListener directly and having a number of no-op methods.
 *
 * @author Alex Black
 */
public interface TrainingListener {

    /**
     * Event listener for each iteration. Called once, after each parameter update has ocurred while training the network
     * @param iteration the iteration
     * @param model the model iterating
     */
    void iterationDone(Model model, int iteration, int epoch);

    /**
     * Called once at the start of each epoch, when using methods such as {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#fit(DataSetIterator)},
     * {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(DataSetIterator)} or {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(MultiDataSetIterator)}
     */
    void onEpochStart(Model model);

    /**
     * Called once at the end of each epoch, when using methods such as {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#fit(DataSetIterator)},
     * {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(DataSetIterator)} or {@link org.deeplearning4j.nn.graph.ComputationGraph#fit(MultiDataSetIterator)}
     */
    void onEpochEnd(Model model);

    /**
     * Called once per iteration (forward pass) for activations (usually for a {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork}),
     * only at training time
     *
     * @param model       Model
     * @param activations Layer activations (including input)
     */
    void onForwardPass(Model model, List<INDArray> activations);

    /**
     * Called once per iteration (forward pass) for activations (usually for a {@link org.deeplearning4j.nn.graph.ComputationGraph}),
     * only at training time
     *
     * @param model       Model
     * @param activations Layer activations (including input)
     */
    void onForwardPass(Model model, Map<String, INDArray> activations);


    /**
     * Called once per iteration (backward pass) <b>before the gradients are updated</b>
     * Gradients are available via {@link Model#gradient()}.
     * Note that gradients will likely be updated in-place - thus they should be copied or processed synchronously
     * in this method.
     * <p>
     * For updates (gradients post learning rate/momentum/rmsprop etc) see {@link #onBackwardPass(Model)}
     *
     * @param model Model
     */
    void onGradientCalculation(Model model);

    /**
     * Called once per iteration (backward pass) after gradients have been calculated, and updated
     * Gradients are available via {@link Model#gradient()}.
     * <p>
     * Unlike {@link #onGradientCalculation(Model)} the gradients at this point will be post-update, rather than
     * raw (pre-update) gradients at that method call.
     *
     * @param model Model
     */
    void onBackwardPass(Model model);

}
