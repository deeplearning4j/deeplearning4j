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

package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.*;

/**
 * {@link NormalizerSerializerStrategy}
 * for {@link ImagePreProcessingScaler}
 *
 * Saves the min range, max range, and max pixel value as
 * doubles
 *
 *
 * @author Adam Gibson
 */
public class ImagePreProcessingSerializerStrategy implements NormalizerSerializerStrategy<ImagePreProcessingScaler> {
    @Override
    public void write(ImagePreProcessingScaler normalizer, OutputStream stream) throws IOException {
        try(DataOutputStream dataOutputStream = new DataOutputStream(stream)) {
            dataOutputStream.writeDouble(normalizer.getMinRange());
            dataOutputStream.writeDouble(normalizer.getMaxRange());
            dataOutputStream.writeDouble(normalizer.getMaxPixelVal());
            dataOutputStream.flush();
        }
    }

    @Override
    public ImagePreProcessingScaler restore(InputStream stream) throws IOException {
        DataInputStream dataOutputStream = new DataInputStream(stream);
        double minRange = dataOutputStream.readDouble();
        double maxRange = dataOutputStream.readDouble();
        double maxPixelVal = dataOutputStream.readDouble();
        ImagePreProcessingScaler ret =  new ImagePreProcessingScaler(minRange,maxRange);
        ret.setMaxPixelVal(maxPixelVal);
        return ret;
    }

    @Override
    public NormalizerType getSupportedType() {
        return NormalizerType.IMAGE_MIN_MAX;
    }
}
