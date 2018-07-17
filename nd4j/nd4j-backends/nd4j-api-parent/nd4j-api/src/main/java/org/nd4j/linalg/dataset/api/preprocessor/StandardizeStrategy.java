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

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;

/**
 * {@link NormalizerStrategy} implementation that will standardize and de-standardize data arrays, based on statistics
 * of the means and standard deviations of the population
 *
 * @author Ede Meijer
 */
@EqualsAndHashCode
public class StandardizeStrategy implements NormalizerStrategy<DistributionStats> {
    /**
     * Normalize a data array
     *
     * @param array the data to normalize
     * @param stats statistics of the data population
     */
    @Override
    public void preProcess(INDArray array, INDArray maskArray, DistributionStats stats) {
        if (array.rank() <= 2) {
            array.subiRowVector(stats.getMean());
            array.diviRowVector(filteredStd(stats));
        }
        // if array Rank is 3 (time series) samplesxfeaturesxtimesteps
        // if array Rank is 4 (images) samplesxchannelsxrowsxcols
        // both cases operations should be carried out in dimension 1
        else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(array, stats.getMean(), array, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(array, filteredStd(stats), array, 1));
        }

        if (maskArray != null) {
            DataSetUtil.setMaskedValuesToZero(array, maskArray);
        }
    }

    /**
     * Denormalize a data array
     *
     * @param array the data to denormalize
     * @param stats statistics of the data population
     */
    @Override
    public void revert(INDArray array, INDArray maskArray, DistributionStats stats) {
        if (array.rank() <= 2) {
            array.muliRowVector(filteredStd(stats));
            array.addiRowVector(stats.getMean());
        } else {
            Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(array, filteredStd(stats), array, 1));
            Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(array, stats.getMean(), array, 1));
        }

        if (maskArray != null) {
            DataSetUtil.setMaskedValuesToZero(array, maskArray);
        }
    }

    /**
     * Create a new {@link NormalizerStats.Builder} instance that can be used to fit new data and of the opType that
     * belongs to the current NormalizerStrategy implementation
     *
     * @return the new builder
     */
    @Override
    public NormalizerStats.Builder newStatsBuilder() {
        return new DistributionStats.Builder();
    }

    private static INDArray filteredStd(DistributionStats stats) {
        /*
            To avoid division by zero when the std deviation is zero, replace zeros by one
         */
        INDArray stdCopy = stats.getStd();
        BooleanIndexing.replaceWhere(stdCopy, 1.0, Conditions.equals(0));
        return stdCopy;
    }
}
