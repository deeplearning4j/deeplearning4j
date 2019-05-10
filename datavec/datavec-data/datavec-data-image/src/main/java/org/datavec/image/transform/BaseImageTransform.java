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

package org.datavec.image.transform;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.bytedeco.javacv.FrameConverter;
import org.datavec.image.data.ImageWritable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.Random;

/**
 *
 * Implements the ImageTransform interface by providing its subclasses
 * with a random object to use in the case of random transformations.
 *
 * @author saudet
 */
@NoArgsConstructor
@JsonIgnoreProperties({"converter", "currentImage"})
@Data
public abstract class BaseImageTransform<F> implements ImageTransform {

    protected Random random;
    protected FrameConverter<F> converter;
    protected ImageWritable currentImage;

    protected BaseImageTransform(Random random) {
        this.random = random;
    }

    @Override
    public final ImageWritable transform(ImageWritable image) {
        return transform(image, random);
    }

    @Override
    public final ImageWritable transform(ImageWritable image, Random random) {
        return currentImage = doTransform(image, random);
    }

    protected abstract ImageWritable doTransform(ImageWritable image, Random random);

    @Override
    public float[] query(float... coordinates) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ImageWritable getCurrentImage() {
        return currentImage;
    }
}
