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
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.dataset.api.preprocessor.stats.MinMaxStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

/**
 * Pre processor for MultiDataSet that normalizes feature values (and optionally label values) to lie between a minimum
 * and maximum value (by default between 0 and 1)
 *
 * @author Ede Meijer
 */
public class MultiNormalizerMinMaxScaler extends AbstractMultiDataSetNormalizer<MinMaxStats> {
    public MultiNormalizerMinMaxScaler() {
        this(0.0, 1.0);
    }

    /**
     * Preprocessor can take a range as minRange and maxRange
     *
     * @param minRange the target range lower bound
     * @param maxRange the target range upper bound
     */
    public MultiNormalizerMinMaxScaler(double minRange, double maxRange) {
        super(new MinMaxStrategy(minRange, maxRange));
    }

    public double getTargetMin() {
        return ((MinMaxStrategy) strategy).getMinRange();
    }

    public double getTargetMax() {
        return ((MinMaxStrategy) strategy).getMaxRange();
    }

    @Override
    protected NormalizerStats.Builder newBuilder() {
        return new MinMaxStats.Builder();
    }

    public INDArray getMin(int input) {
        return getFeatureStats(input).getLower();
    }

    public INDArray getMax(int input) {
        return getFeatureStats(input).getUpper();
    }

    public INDArray getLabelMin(int output) {
        return getLabelStats(output).getLower();
    }

    public INDArray getLabelMax(int output) {
        return getLabelStats(output).getUpper();
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.MULTI_MIN_MAX;
    }
}
