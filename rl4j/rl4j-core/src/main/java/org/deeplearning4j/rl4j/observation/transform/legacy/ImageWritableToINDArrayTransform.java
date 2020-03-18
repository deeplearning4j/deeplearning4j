/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.observation.transform.legacy;

import org.datavec.api.transform.Operation;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;

public class ImageWritableToINDArrayTransform implements Operation<ImageWritable, INDArray> {

    private final int height;
    private final int width;
    private final NativeImageLoader loader;

    public ImageWritableToINDArrayTransform(int height, int width) {
        this.height = height;
        this.width = width;
        this.loader = new NativeImageLoader(height, width);
    }

    @Override
    public INDArray transform(ImageWritable imageWritable) {
        INDArray out = null;
        try {
            out = loader.asMatrix(imageWritable);
        } catch (IOException e) {
            e.printStackTrace();
        }
        out = out.reshape(1, height, width);
        INDArray compressed = out.castTo(DataType.UINT8);
        return compressed;
    }
}
