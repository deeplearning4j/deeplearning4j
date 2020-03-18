/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.observation.transform.operation.historymerge;

import org.deeplearning4j.rl4j.observation.transform.operation.HistoryMergeTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A HistoryMergeAssembler is used with the {@link HistoryMergeTransform HistoryMergeTransform}. This interface defines how the array of INDArray
 * given by the {@link HistoryMergeElementStore HistoryMergeElementStore} is packaged into the single INDArray that will be
 * returned by the HistoryMergeTransform
 *
 * @author Alexandre Boulanger
 */
public interface HistoryMergeAssembler {
    /**
     * Assemble an array of INDArray into a single INArray
     * @param elements The input INDArray[]
     * @return the assembled INDArray
     */
    INDArray assemble(INDArray[] elements);
}
