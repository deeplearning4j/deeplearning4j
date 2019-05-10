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
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

import java.io.File;
import java.io.IOException;

/**
 * Created by susaneraly, Ede Meijer
 * variance and mean
 * Pre processor for DataSet that normalizes feature values (and optionally label values) to have 0 mean and a standard
 * deviation of 1
 */
@EqualsAndHashCode(callSuper = true)
public class NormalizerStandardize extends AbstractDataSetNormalizer<DistributionStats> {
    public NormalizerStandardize(@NonNull INDArray featureMean, @NonNull INDArray featureStd) {
        this();
        setFeatureStats(new DistributionStats(featureMean, featureStd));
        fitLabel(false);
    }

    public NormalizerStandardize(@NonNull INDArray featureMean, @NonNull INDArray featureStd,
                    @NonNull INDArray labelMean, @NonNull INDArray labelStd) {
        this();
        setFeatureStats(new DistributionStats(featureMean, featureStd));
        setLabelStats(new DistributionStats(labelMean, labelStd));
        fitLabel(true);
    }

    public NormalizerStandardize() {
        super(new StandardizeStrategy());
    }

    public void setLabelStats(@NonNull INDArray labelMean, @NonNull INDArray labelStd) {
        setLabelStats(new DistributionStats(labelMean, labelStd));
    }

    public INDArray getMean() {
        return getFeatureStats().getMean();
    }

    public INDArray getLabelMean() {
        return getLabelStats().getMean();
    }

    public INDArray getStd() {
        return getFeatureStats().getStd();
    }

    public INDArray getLabelStd() {
        return getLabelStats().getStd();
    }

    /**
     * Load the means and standard deviations from the file system
     *
     * @param files the files to load from. Needs 4 files if normalizing labels, otherwise 2.
     */
    public void load(File... files) throws IOException {
        setFeatureStats(DistributionStats.load(files[0], files[1]));
        if (isFitLabel()) {
            setLabelStats(DistributionStats.load(files[2], files[3]));
        }
    }

    /**
     * @param files the files to save to. Needs 4 files if normalizing labels, otherwise 2.
     * @deprecated use {@link NormalizerSerializer} instead
     * <p>
     * Save the current means and standard deviations to the file system
     */
    public void save(File... files) throws IOException {
        getFeatureStats().save(files[0], files[1]);
        if (isFitLabel()) {
            getLabelStats().save(files[2], files[3]);
        }
    }

    @Override
    protected NormalizerStats.Builder newBuilder() {
        return new DistributionStats.Builder();
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.STANDARDIZE;
    }
}
