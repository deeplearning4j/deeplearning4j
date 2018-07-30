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

package org.deeplearning4j.nn.api;

import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * @author raver119
 */
public interface NeuralNetwork {

    /**
     * This method does initialization of model
     *
     * PLEASE NOTE: All implementations should track own state, to avoid double spending
     */
    void init();

    /**
     * This method returns model parameters as single INDArray
     *
     * @return
     */
    INDArray params();

    /**
     * This method returns updater state (if applicable), null otherwise
     * @return
     */
    INDArray updaterState();

    /**
     * This method returns Optimizer used for training
     *
     * @return
     */
    ConvexOptimizer getOptimizer();

    /**
     * This method fits model with a given DataSet
     *
     * @param dataSet
     */
    void fit(DataSet dataSet);

    /**
     * This method fits model with a given MultiDataSet
     *
     * @param dataSet
     */
    void fit(MultiDataSet dataSet);

    /**
     * This method fits model with a given DataSetIterator
     *
     * @param iterator
     */
    void fit(DataSetIterator iterator);

    /**
     * This method fits model with a given MultiDataSetIterator
     *
     * @param iterator
     */
    void fit(MultiDataSetIterator iterator);

    /**
     * This method executes evaluation of the model against given iterator and evaluation implementations
     *
     * @param iterator
     */
    <T extends IEvaluation> T[] doEvaluation(DataSetIterator iterator, T... evaluations);

    /**
     * This method executes evaluation of the model against given iterator and evaluation implementations
     *
     * @param iterator
     */
    <T extends IEvaluation> T[] doEvaluation(MultiDataSetIterator iterator, T... evaluations);
}
