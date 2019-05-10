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
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MultiStandardizeSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.dataset.api.preprocessor.stats.DistributionStats;
import org.nd4j.linalg.dataset.api.preprocessor.stats.NormalizerStats;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Pre processor for MultiDataSet that normalizes feature values (and optionally label values) to have 0 mean and
 * a standard deviation of 1
 *
 * @author Ede Meijer
 */
@EqualsAndHashCode(callSuper = true)
public class MultiNormalizerStandardize extends AbstractMultiDataSetNormalizer<DistributionStats> {
    public MultiNormalizerStandardize() {
        super(new StandardizeStrategy());
    }

    @Override
    protected NormalizerStats.Builder newBuilder() {
        return new DistributionStats.Builder();
    }

    public INDArray getFeatureMean(int input) {
        return getFeatureStats(input).getMean();
    }

    public INDArray getLabelMean(int output) {
        return getLabelStats(output).getMean();
    }

    public INDArray getFeatureStd(int input) {
        return getFeatureStats(input).getStd();
    }

    public INDArray getLabelStd(int output) {
        return getLabelStats(output).getStd();
    }

    /**
     * Load means and standard deviations from the file system
     *
     * @param featureFiles source files for features, requires 2 files per input, alternating mean and stddev files
     * @param labelFiles   source files for labels, requires 2 files per output, alternating mean and stddev files
     */
    public void load(@NonNull List<File> featureFiles, @NonNull List<File> labelFiles) throws IOException {
        setFeatureStats(load(featureFiles));
        if (isFitLabel()) {
            setLabelStats(load(labelFiles));
        }
    }

    private List<DistributionStats> load(List<File> files) throws IOException {
        ArrayList<DistributionStats> stats = new ArrayList<>(files.size() / 2);
        for (int i = 0; i < files.size() / 2; i++) {
            stats.add(DistributionStats.load(files.get(i * 2), files.get(i * 2 + 1)));
        }
        return stats;
    }

    /**
     * @param featureFiles target files for features, requires 2 files per input, alternating mean and stddev files
     * @param labelFiles   target files for labels, requires 2 files per output, alternating mean and stddev files
     * @deprecated use {@link MultiStandardizeSerializerStrategy} instead
     * <p>
     * Save the current means and standard deviations to the file system
     */
    public void save(@NonNull List<File> featureFiles, @NonNull List<File> labelFiles) throws IOException {
        saveStats(getFeatureStats(), featureFiles);
        if (isFitLabel()) {
            saveStats(getLabelStats(), labelFiles);
        }
    }

    private void saveStats(List<DistributionStats> stats, List<File> files) throws IOException {
        int requiredFiles = stats.size() * 2;
        if (requiredFiles != files.size()) {
            throw new RuntimeException(String.format("Need twice as many files as inputs / outputs (%d), got %d",
                            requiredFiles, files.size()));
        }

        for (int i = 0; i < stats.size(); i++) {
            stats.get(i).save(files.get(i * 2), files.get(i * 2 + 1));
        }
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.MULTI_STANDARDIZE;
    }
}
