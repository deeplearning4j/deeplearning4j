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

package org.deeplearning4j.eval;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;
import java.util.List;

/**
 * A general purpose interface for evaluating neural networks - methods are shared by implemetations such as
 * {@link Evaluation}, {@link RegressionEvaluation}, {@link ROC}, {@link ROCMultiClass}
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY)
public interface IEvaluation<T extends IEvaluation> extends Serializable {


    /**
     *
     * @param labels
     * @param networkPredictions
     */
    void eval(INDArray labels, INDArray networkPredictions);

    /**
     *
     * @param labels
     * @param networkPredictions
     * @param recordMetaData
     */
    void eval(INDArray labels, INDArray networkPredictions, List<? extends Serializable> recordMetaData);

    /**
     *
     * @param labels
     * @param networkPredictions
     * @param maskArray
     */
    void eval(INDArray labels, INDArray networkPredictions, INDArray maskArray);


    /**
     *
     * @param labels
     * @param predicted
     */
    void evalTimeSeries(INDArray labels, INDArray predicted);

    /**
     *
     * @param labels
     * @param predicted
     * @param labelsMaskArray
     */
    void evalTimeSeries(INDArray labels, INDArray predicted, INDArray labelsMaskArray);

    /**
     *
     * @param other
     */
    void merge(T other);

    /**
     *
     */
    void reset();

    /**
     *
     * @return
     */
    String stats();

    /**
     *
     * @return
     */
    String toJson();

    /**
     *
     * @return
     */
    String toYaml();
}
