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

package org.deeplearning4j.nn.api.layers;

import org.deeplearning4j.nn.api.Classifier;
import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

/**
 * Interface for output layers (those that calculate gradients with respect to a labels array)
 */
public interface IOutputLayer extends Layer, Classifier {

    /**
     * Returns true if labels are required
     * for this output layer
     * @return true if this output layer needs labels or not
     */
    boolean needsLabels();

    /**
     * Set the labels array for this output layer
     *
     * @param labels Labels array to set
     */
    void setLabels(INDArray labels);

    /**
     * Get the labels array previously set with {@link #setLabels(INDArray)}
     *
     * @return Labels array, or null if it has not been set
     */
    INDArray getLabels();

    /**
     * Compute score after labels and input have been set.
     *
     * @param fullNetworkRegScore Regularization score (l1/l2/weight decay) for the entire network
     * @param training            whether score should be calculated at train or test time (this affects things like application of
     *                            dropout, etc)
     * @return score (loss function)
     */
    double computeScore(double fullNetworkRegScore, boolean training, LayerWorkspaceMgr workspaceMgr);

    /**
     * Compute the score for each example individually, after labels and input have been set.
     *
     * @param fullNetworkRegScore Regularization score (l1/l2/weight decay) for the entire network
     * @return A column INDArray of shape [numExamples,1], where entry i is the score of the ith example
     */
    INDArray computeScoreForExamples(double fullNetworkRegScore, LayerWorkspaceMgr workspaceMgr);


}
