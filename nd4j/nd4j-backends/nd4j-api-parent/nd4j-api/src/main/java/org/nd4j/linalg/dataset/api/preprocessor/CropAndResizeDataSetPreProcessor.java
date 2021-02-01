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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

/**
 * The CropAndResizeDataSetPreProcessor will crop and resize the processed dataset.
 * NOTE: The data format must be NHWC
 *
 * @author Alexandre Boulanger
 */
public class CropAndResizeDataSetPreProcessor implements DataSetPreProcessor {

    public enum ResizeMethod {
        Bilinear,
        NearestNeighbor
    }

    private final long[] resizedShape;
    private final INDArray indices;
    private final INDArray resize;
    private final INDArray boxes;
    private final int method;

    /**
     *
     * @param originalHeight Height of the input datasets
     * @param originalWidth Width of the input datasets
     * @param cropYStart y coord of the starting point on the input datasets
     * @param cropXStart x coord of the starting point on the input datasets
     * @param resizedHeight Height of the output dataset
     * @param resizedWidth Width of the output dataset
     * @param numChannels
     * @param resizeMethod
     */
    public CropAndResizeDataSetPreProcessor(int originalHeight, int originalWidth, int cropYStart, int cropXStart, int resizedHeight, int resizedWidth, int numChannels, ResizeMethod resizeMethod) {
        Preconditions.checkArgument(originalHeight > 0, "originalHeight must be greater than 0, got %s", originalHeight);
        Preconditions.checkArgument(originalWidth > 0, "originalWidth must be greater than 0, got %s", originalWidth);
        Preconditions.checkArgument(cropYStart >= 0, "cropYStart must be greater or equal to 0, got %s", cropYStart);
        Preconditions.checkArgument(cropXStart >= 0, "cropXStart must be greater or equal to 0, got %s", cropXStart);
        Preconditions.checkArgument(resizedHeight > 0, "resizedHeight must be greater than 0, got %s", resizedHeight);
        Preconditions.checkArgument(resizedWidth > 0, "resizedWidth must be greater than 0, got %s", resizedWidth);
        Preconditions.checkArgument(numChannels > 0, "numChannels must be greater than 0, got %s", numChannels);

        resizedShape = new long[] { 1, resizedHeight, resizedWidth, numChannels };

        boxes = Nd4j.create(new float[] {
                (float)cropYStart / (float)originalHeight,
                (float)cropXStart / (float)originalWidth,
                (float)(cropYStart + resizedHeight) / (float)originalHeight,
                (float)(cropXStart + resizedWidth) / (float)originalWidth
            }, new long[] { 1, 4 }, DataType.FLOAT);
        indices = Nd4j.create(new int[] { 0 }, new long[] { 1, 1 }, DataType.INT);

        resize = Nd4j.create(new int[] { resizedHeight, resizedWidth }, new long[] { 1, 2 }, DataType.INT);
        method = resizeMethod == ResizeMethod.Bilinear ? 0 : 1;
    }

    /**
     * NOTE: The data format must be NHWC
     */
    @Override
    public void preProcess(DataSet dataSet) {
        Preconditions.checkNotNull(dataSet, "Encountered null dataSet");

        if(dataSet.isEmpty()) {
            return;
        }

        INDArray input = dataSet.getFeatures();
        INDArray output = Nd4j.create(LongShapeDescriptor.fromShape(resizedShape, input.dataType()), false);

        CustomOp op = DynamicCustomOp.builder("crop_and_resize")
                .addInputs(input, boxes, indices, resize)
                .addIntegerArguments(method)
                .addOutputs(output)
                .build();
        Nd4j.getExecutioner().exec(op);

        dataSet.setFeatures(output);
    }
}
