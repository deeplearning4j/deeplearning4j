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

import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.api.transform.Operation;
import org.datavec.image.data.ImageWritable;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.bytedeco.opencv.global.opencv_core.CV_32FC;

public class EncodableToImageWritableTransform implements Operation<Encodable, ImageWritable> {

    private final OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
    private final int height;
    private final int width;
    private final int colorChannels;

    public EncodableToImageWritableTransform(int height, int width, int colorChannels) {
        this.height = height;
        this.width = width;
        this.colorChannels = colorChannels;
    }

    @Override
    public ImageWritable transform(Encodable encodable) {
        INDArray indArray = Nd4j.create((encodable).toArray()).reshape(height, width, colorChannels);
        Mat mat = new Mat(height, width, CV_32FC(3), indArray.data().pointer());
        return new ImageWritable(converter.convert(mat));
    }

}
