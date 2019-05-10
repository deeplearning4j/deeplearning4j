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
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;

/**
 * Created by susaneraly on 6/23/16.
 * A preprocessor specifically for images that applies min max scaling
 * Can take a range, so pixel values can be scaled from 0->255 to minRange->maxRange
 * default minRange = 0 and maxRange = 1;
 * If pixel values are not 8 bits, you can specify the number of bits as the third argument in the constructor
 * For values that are already floating point, specify the number of bits as 1
 *
 */
@Slf4j
@Getter
@Setter
@EqualsAndHashCode
public class ImagePreProcessingScaler implements DataNormalization {

    private double minRange, maxRange;
    private double maxPixelVal;
    private int maxBits;

    public ImagePreProcessingScaler() {
        this(0, 1, 8);
    }

    public ImagePreProcessingScaler(double a, double b) {
        this(a, b, 8);
    }

    /**
     * Preprocessor can take a range as minRange and maxRange
     * @param a, default = 0
     * @param b, default = 1
     * @param maxBits in the image, default = 8
     */
    public ImagePreProcessingScaler(double a, double b, int maxBits) {
        //Image values are not always from 0 to 255 though
        //some images are 16-bit, some 32-bit, integer, or float, and those BTW already come with values in [0..1]...
        //If the max expected value is 1, maxBits should be specified as 1
        maxPixelVal = Math.pow(2, maxBits) - 1;
        this.minRange = a;
        this.maxRange = b;
    }

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
        features.divi(this.maxPixelVal); //Scaled to 0->1
        if (this.maxRange - this.minRange != 1)
            features.muli(this.maxRange - this.minRange); //Scaled to minRange -> maxRange
        if (this.minRange != 0)
            features.addi(this.minRange); //Offset by minRange
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
        return NormalizerType.IMAGE_MIN_MAX;
    }

    @Override
    public void revertFeatures(INDArray features) {
        if (minRange != 0) {
            features.subi(minRange);
        }
        if (maxRange - minRange != 1.0) {
            features.divi(maxRange - minRange);
        }
        features.muli(this.maxPixelVal);
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
