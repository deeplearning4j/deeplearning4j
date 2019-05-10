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

import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

/**
 * A DataSetPreProcessor used to flatten a 4d CNN features array to a flattened 2d format (for use in networks such
 * as a DenseLayer/multi-layer perceptron)
 *
 * @author Alex Black
 */
public class ImageFlatteningDataSetPreProcessor implements DataSetPreProcessor {
    @Override
    public void preProcess(DataSet toPreProcess) {
        INDArray input = toPreProcess.getFeatures();
        if (input.rank() == 2)
            return; //No op: should usually never happen in a properly configured data pipeline

        //Assume input is standard rank 4 activations - i.e., CNN image data
        //First: we require input to be in c order. But c order (as declared in array order) isn't enough; also need strides to be correct
        if (input.ordering() != 'c' || !Shape.strideDescendingCAscendingF(input))
            input = input.dup('c');

        val inShape = input.shape(); //[miniBatch,depthOut,outH,outW]
        val outShape = new long[] {inShape[0], inShape[1] * inShape[2] * inShape[3]};

        INDArray reshaped = input.reshape('c', outShape);
        toPreProcess.setFeatures(reshaped);
    }
}
