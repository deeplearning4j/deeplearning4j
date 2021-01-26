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

package org.nd4j.evaluation.custom;

import java.io.Serializable;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A lambda used to get an evaluation result for a batch
 * See {@link CustomEvaluation}
 */
public interface EvaluationLambda<T> {
    public T eval(INDArray labels, INDArray networkPredictions, INDArray maskArray,
                  List<? extends Serializable> recordMetaData);
}
