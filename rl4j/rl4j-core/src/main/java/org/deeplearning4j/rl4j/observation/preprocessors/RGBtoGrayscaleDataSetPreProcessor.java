/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.rl4j.observation.preprocessors;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * The RGBtoGrayscaleDataSetPreProcessor will turn a DataSet of a RGB image into a grayscale one.
 * NOTE: Expects data format to be NCHW. After processing, the channel dimension is eliminated. (NCHW -> NHW)
 *
 * @author Alexandre Boulanger
 */
public class RGBtoGrayscaleDataSetPreProcessor implements DataSetPreProcessor {

    private static final float RED_RATIO = 0.3f;
    private static final float GREEN_RATIO = 0.59f;
    private static final float BLUE_RATIO = 0.11f;

    @Override
    public void preProcess(DataSet dataSet) {
        Preconditions.checkNotNull(dataSet, "Encountered null dataSet");

        if(dataSet.isEmpty()) {
            return;
        }

        // Extract channels
        INDArray features = dataSet.getFeatures().slice(0, 0);
        INDArray R = features.slice(0, 0);
        INDArray G = features.slice(1, 0);
        INDArray B = features.slice(2, 0);

        R.muli(RED_RATIO);
        G.muli(GREEN_RATIO);
        B.muli(BLUE_RATIO);

        R.addi(G).addi(B);

        // Input shape is { 1, c, h, w }; output is { 1, h, w }
        long[] sliceShape = R.shape();
        INDArray result = Nd4j.create(new long[] { 1, sliceShape[0], sliceShape[1] });
        result.putSlice(0, R);

        dataSet.setFeatures(result);
    }
}
