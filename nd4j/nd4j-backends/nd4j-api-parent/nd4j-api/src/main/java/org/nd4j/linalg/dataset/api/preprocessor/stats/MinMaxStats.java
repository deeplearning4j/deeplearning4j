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

package org.nd4j.linalg.dataset.api.preprocessor.stats;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 * Statistics about the lower bounds and upper bounds of values in data.
 * Can be constructed incrementally by using the DynamicCustomOpsBuilder,
 * which is useful for obtaining these statistics from an
 * iterator.
 *
 * @author Ede Meijer
 */
@EqualsAndHashCode
@Slf4j
public class MinMaxStats implements NormalizerStats {

    @Getter
    private final INDArray lower;
    @Getter
    private final INDArray upper;
    private INDArray range;

    /**
     * @param lower row vector of lower bounds
     * @param upper row vector of upper bounds
     */
    public MinMaxStats(@NonNull INDArray lower, @NonNull INDArray upper) {
        // Check for 0 differences and round up to epsilon
        INDArray diff = upper.sub(lower);
        INDArray addedPadding = Transforms.max(diff, Nd4j.EPS_THRESHOLD).subi(diff);
        // If any entry in `addedPadding` is not 0, then we had to add something to prevent 0 difference, Add this same
        // value to the upper bounds to actually apply the padding, and log about it
        if (addedPadding.sumNumber().doubleValue() > 0) {
            log.info("API_INFO: max val minus min val found to be zero. Transform will round up to epsilon to avoid nans.");
            upper.addi(addedPadding);
        }

        this.lower = lower;
        this.upper = upper;
    }

    /**
     * Get the feature wise
     * range for the statistics.
     * Note that this is a lazy getter.
     * It is only computed when needed.
     * @return the feature wise range
     * given the min and max
     */
    public INDArray getRange() {
        if (range == null) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                range = upper.sub(lower);
            }
        }
        return range;
    }

    /**
     * DynamicCustomOpsBuilder class that can incrementally update a running lower and upper bound in order to create statistics for a
     * large set of data
     */
    public static class Builder implements NormalizerStats.Builder<MinMaxStats> {
        private INDArray runningLower;
        private INDArray runningUpper;

        /**
         * Add the features of a DataSet to the statistics
         */
        public MinMaxStats.Builder addFeatures(@NonNull org.nd4j.linalg.dataset.api.DataSet dataSet) {
            return add(dataSet.getFeatures(), dataSet.getFeaturesMaskArray());
        }

        /**
         * Add the labels of a DataSet to the statistics
         */
        public MinMaxStats.Builder addLabels(@NonNull org.nd4j.linalg.dataset.api.DataSet dataSet) {
            return add(dataSet.getLabels(), dataSet.getLabelsMaskArray());
        }

        /**
         * Add rows of data to the statistics
         *
         * @param data the matrix containing multiple rows of data to include
         * @param mask (optionally) the mask of the data, useful for e.g. time series
         */
        public MinMaxStats.Builder add(@NonNull INDArray data, INDArray mask) {
            data = DataSetUtil.tailor2d(data, mask);
            if (data == null) {
                // Nothing to add. Either data is empty or completely masked. Just skip it, otherwise we will get
                // null pointer exceptions.
                return this;
            }

            INDArray tad = data.javaTensorAlongDimension(0, 0);
            INDArray batchMin = data.min(0);
            INDArray batchMax = data.max(0);
            if (!Arrays.equals(batchMin.shape(), batchMax.shape()))
                throw new IllegalStateException(
                                "Data min and max must be same shape. Likely a bug in the operation changing the input?");
            if (runningLower == null) {
                // First batch
                // Create copies because min and max are views to the same data set, which will cause problems with the
                // side effects of Transforms.min and Transforms.max
                runningLower = batchMin.dup();
                runningUpper = batchMax.dup();
            } else {
                // Update running bounds
                Transforms.min(runningLower, batchMin, false);
                Transforms.max(runningUpper, batchMax, false);
            }

            return this;
        }

        /**
         * Create a DistributionStats object from the data ingested so far. Can be used multiple times when updating
         * online.
         */
        public MinMaxStats build() {
            if (runningLower == null) {
                throw new RuntimeException("No data was added, statistics cannot be determined");
            }
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                return new MinMaxStats(runningLower.dup(), runningUpper.dup());
            }
        }
    }
}
