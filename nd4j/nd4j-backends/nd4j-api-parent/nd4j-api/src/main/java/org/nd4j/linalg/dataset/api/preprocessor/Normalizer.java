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

package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;

/**
 * Base interface for all normalizers
 *
 * @param <T> either {@link DataSet} or {@link MultiDataSet}
 */
public interface Normalizer<T> {
    /**
     * Fit a dataset (only compute based on the statistics from this dataset)
     *
     * @param dataSet the dataset to compute on
     */
    void fit(T dataSet);

    /**
     * Transform the dataset
     *
     * @param toPreProcess the dataset to re process
     */
    void transform(T toPreProcess);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance (arrays are modified in-place)
     *
     * @param toRevert DataSet to revert the normalization on
     */
    void revert(T toRevert);

    /**
     * Get the enum opType of this normalizer
     *
     * @return the opType
     * @see NormalizerSerializerStrategy#getSupportedType()
     */
    NormalizerType getType();
}
