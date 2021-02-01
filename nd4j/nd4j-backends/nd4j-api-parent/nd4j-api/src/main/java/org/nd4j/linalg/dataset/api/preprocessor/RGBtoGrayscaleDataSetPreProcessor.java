/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset.api.preprocessor;

import org.nd4j.common.base.Preconditions;
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

        INDArray originalFeatures = dataSet.getFeatures();
        long[] originalShape = originalFeatures.shape();

        // result shape is NHW
        INDArray result = Nd4j.create(originalShape[0], originalShape[2], originalShape[3]);

        for(long n = 0, numExamples = originalShape[0]; n < numExamples; ++n) {
            // Extract channels
            INDArray itemFeatures = originalFeatures.slice(n, 0); // shape is CHW
            INDArray R = itemFeatures.slice(0, 0);  // shape is HW
            INDArray G = itemFeatures.slice(1, 0);
            INDArray B = itemFeatures.slice(2, 0);

            // Convert
            R.muli(RED_RATIO);
            G.muli(GREEN_RATIO);
            B.muli(BLUE_RATIO);
            R.addi(G).addi(B);

            result.putSlice((int)n, R);
        }

        dataSet.setFeatures(result);
    }
}
