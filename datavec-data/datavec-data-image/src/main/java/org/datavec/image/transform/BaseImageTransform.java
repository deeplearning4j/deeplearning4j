/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */
package org.datavec.image.transform;

import lombok.NoArgsConstructor;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.FrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.data.ImageWritable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.Map;
import java.util.Random;

/**
 *
 * Implements the ImageTransform interface by providing its subclasses
 * with a random object to use in the case of random transformations.
 *
 * @author saudet
 */
@NoArgsConstructor
@JsonIgnoreProperties({"safeConverter"})
public abstract class BaseImageTransform<F> implements ImageTransform {

    protected Random random;
    protected Map<Long, FrameConverter<F>> safeConverter;

    protected BaseImageTransform(Random random) {
        this.random = random;
    }

    @Override
    public ImageWritable transform(ImageWritable image) {
        return transform(image, random);
    }

    abstract FrameConverter<F> getSafeConverter(long threadId);

    /**
     * Returns thread-safe Mat FrameConverter
     * the purpose of this method is to reduce code redundancies for image transforms
     * @param threadId
     * @return
     */
    protected FrameConverter<opencv_core.Mat> getSafeMatConverter(long threadId) {
        if (safeConverter.containsKey(threadId)) {
            return (FrameConverter<opencv_core.Mat>) safeConverter.get(Thread.currentThread().getId());
        }else {
            FrameConverter<opencv_core.Mat> converter = new OpenCVFrameConverter.ToMat();
            safeConverter.put(threadId, (FrameConverter<F>) converter);
            return converter;
        }
    }
}
