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

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This is a preprocessor specifically for VGG16.
 * It subtracts the mean RGB value, computed on the training set, from each pixel as reported in:
 * https://arxiv.org/pdf/1409.1556.pdf
 * @author susaneraly
 */
@Slf4j
public class VGG16ImagePreProcessor implements DataNormalization {

    public static final INDArray VGG_MEAN_OFFSET_BGR = Nd4j.create(new double[] {123.68, 116.779, 103.939});

    /**
     * Fit a dataset (only compute
     * based on the statistics from this dataset0
     *
     * @param dataSet the dataset to compute on
     */
    @Override
    public void fit(DataSet dataSet) {

    }

    /**
     * Iterates over a dataset
     * accumulating statistics for normalization
     *
     * @param iterator the iterator to use for
     *                 collecting statistics.
     */
    @Override
    public void fit(DataSetIterator iterator) {

    }

    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray features = toPreProcess.getFeatures();
        this.preProcess(features);
    }

    public void preProcess(INDArray features) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(features.dup(), VGG_MEAN_OFFSET_BGR, features, 1));
    }

    /**
     * Transform the data
     * @param toPreProcess the dataset to transform
     */
    @Override
    public void transform(DataSet toPreProcess) {
        this.preProcess(toPreProcess);
    }

    @Override
    public void transform(INDArray features) {
        this.preProcess(features);
    }

    @Override
    public void transform(INDArray features, INDArray featuresMask) {
        transform(features);
    }

    @Override
    public void transformLabel(INDArray label) {
        //No op
    }

    @Override
    public void transformLabel(INDArray labels, INDArray labelsMask) {
        transformLabel(labels);
    }

    @Override
    public void revert(DataSet toRevert) {
        revertFeatures(toRevert.getFeatures());
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.IMAGE_VGG16;
    }

    @Override
    public void revertFeatures(INDArray features) {
        Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(features.dup(), VGG_MEAN_OFFSET_BGR, features, 1));
    }

    @Override
    public void revertFeatures(INDArray features, INDArray featuresMask) {
        revertFeatures(features);
    }

    @Override
    public void revertLabels(INDArray labels) {
        //No op
    }

    @Override
    public void revertLabels(INDArray labels, INDArray labelsMask) {
        revertLabels(labels);
    }

    @Override
    public void fitLabel(boolean fitLabels) {
        if (fitLabels) {
            log.warn("Labels fitting not currently supported for ImagePreProcessingScaler. Labels will not be modified");
        }
    }

    @Override
    public boolean isFitLabel() {
        return false;
    }
}
