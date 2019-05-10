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
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerType;

/**
 * A preprocessor specifically for images that applies min max scaling to one or more of the feature arrays
 * in a MultiDataSet.<br>
 * Can take a range, so pixel values can be scaled from 0->255 to minRange->maxRange
 * default minRange = 0 and maxRange = 1;
 * If pixel values are not 8 bits, you can specify the number of bits as the third argument in the constructor
 * For values that are already floating point, specify the number of bits as 1
 *
 * @author Alex Black (MultiDataSet version), Susan Eraly (original ImagePreProcessingScaler)
 */
public class ImageMultiPreProcessingScaler implements MultiDataNormalization {


    private double minRange, maxRange;
    private double maxPixelVal;
    private int[] featureIndices;

    public ImageMultiPreProcessingScaler(int... featureIndices) {
        this(0, 1, 8, featureIndices);
    }

    public ImageMultiPreProcessingScaler(double a, double b, int[] featureIndices) {
        this(a, b, 8, featureIndices);
    }

    /**
     * Preprocessor can take a range as minRange and maxRange
     * @param a, default = 0
     * @param b, default = 1
     * @param maxBits in the image, default = 8
     * @param featureIndices Indices of feature arrays to process. If only one feature array is present,
     *                       this should always be 0
     */
    public ImageMultiPreProcessingScaler(double a, double b, int maxBits, int[] featureIndices) {
        if(featureIndices == null || featureIndices.length == 0){
            throw new IllegalArgumentException("Invalid feature indices: the indices of the features arrays to apply "
                    + "the normalizer to must be specified. MultiDataSet/MultiDataSetIterators with only a single feature"
                    + " array, this should be set to 0. Otherwise specify the indexes of all the feature arrays to apply"
                    + " the normalizer to.");
        }
        //Image values are not always from 0 to 255 though
        //some images are 16-bit, some 32-bit, integer, or float, and those BTW already come with values in [0..1]...
        //If the max expected value is 1, maxBits should be specified as 1
        maxPixelVal = Math.pow(2, maxBits) - 1;
        this.minRange = a;
        this.maxRange = b;
        this.featureIndices = featureIndices;
    }

    @Override
    public void fit(MultiDataSetIterator iterator) {
        //No op
    }

    @Override
    public void preProcess(MultiDataSet multiDataSet) {
        for( int i=0; i<featureIndices.length; i++ ){
            INDArray f = multiDataSet.getFeatures(featureIndices[i]);
            f.divi(this.maxPixelVal); //Scaled to 0->1
            if (this.maxRange - this.minRange != 1)
                f.muli(this.maxRange - this.minRange); //Scaled to minRange -> maxRange
            if (this.minRange != 0)
                f.addi(this.minRange); //Offset by minRange
        }
    }

    @Override
    public void revertFeatures(INDArray[] features, INDArray[] featuresMask) {
        revertFeatures(features);
    }

    @Override
    public void revertFeatures(INDArray[] features) {
        for( int i=0; i<featureIndices.length; i++ ){
            INDArray f = features[featureIndices[i]];
            if (minRange != 0) {
                f.subi(minRange);
            }
            if (maxRange - minRange != 1.0) {
                f.divi(maxRange - minRange);
            }
            f.muli(this.maxPixelVal);
        }
    }

    @Override
    public void revertLabels(INDArray[] labels, INDArray[] labelsMask) {
        //No op
    }

    @Override
    public void revertLabels(INDArray[] labels) {
        //No op
    }

    @Override
    public void fit(MultiDataSet dataSet) {
        //No op
    }

    @Override
    public void transform(MultiDataSet toPreProcess) {
        preProcess(toPreProcess);
    }

    @Override
    public void revert(MultiDataSet toRevert) {
        revertFeatures(toRevert.getFeatures(), toRevert.getFeaturesMaskArrays());
    }

    @Override
    public NormalizerType getType() {
        return NormalizerType.IMAGE_MIN_MAX;
    }
}
