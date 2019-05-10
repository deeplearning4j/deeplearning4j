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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * An interface for multi dataset normalizers.
 * Data normalizers compute some sort of statistics
 * over a MultiDataSet and scale the data in some way.
 *
 * @author Ede Meijer
 */
public interface MultiDataNormalization extends Normalizer<MultiDataSet>, MultiDataSetPreProcessor {
    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     *
     * @param iterator the iterator to use for
     *                 collecting statistics.
     */
    void fit(MultiDataSetIterator iterator);

    @Override
    void preProcess(MultiDataSet multiDataSet);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified features array
     *
     * @param features    Features to revert the normalization on
     * @param featuresMask
     */
    void revertFeatures(INDArray[] features, INDArray[] featuresMask);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified features array
     *
     * @param features Features to revert the normalization on
     */
    void revertFeatures(INDArray[] features);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified labels array.
     * If labels normalization is disabled (i.e., {@link #isFitLabel()} == false) then this is a no-op.
     * Can also be used to undo normalization for network output arrays, in the case of regression.
     *
     * @param labels    Labels array to revert the normalization on
     * @param labelsMask Labels mask array (may be null)
     */
    void revertLabels(INDArray[] labels, INDArray[] labelsMask);

    /**
     * Undo (revert) the normalization applied by this DataNormalization instance to the specified labels array.
     * If labels normalization is disabled (i.e., {@link #isFitLabel()} == false) then this is a no-op.
     * Can also be used to undo normalization for network output arrays, in the case of regression.
     *
     * @param labels Labels array to revert the normalization on
     */
    void revertLabels(INDArray[] labels);
}
