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

import org.datavec.image.data.ImageWritable;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.util.Random;

/**
 * Transforms an image in some way, either deterministically or randomly.
 *
 * @author saudet
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.WRAPPER_OBJECT)
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonSubTypes(value = {@JsonSubTypes.Type(value = ColorConversionTransform.class, name = "ColorConversionTransform"),
                @JsonSubTypes.Type(value = BoxImageTransform.class, name = "BoxImageTransform"),
                @JsonSubTypes.Type(value = CropImageTransform.class, name = "CropImageTransform"),
                @JsonSubTypes.Type(value = EqualizeHistTransform.class, name = "EqualizeHistTransform"),
                @JsonSubTypes.Type(value = FilterImageTransform.class, name = "FilterImageTransform"),
                @JsonSubTypes.Type(value = FlipImageTransform.class, name = "FlipImageTransform"),
                @JsonSubTypes.Type(value = LargestBlobCropTransform.class, name = "LargestBlobCropTransform"),
                @JsonSubTypes.Type(value = RandomCropTransform.class, name = "RandomCropTransform"),
                @JsonSubTypes.Type(value = ResizeImageTransform.class, name = "ResizeImageTransform"),
                @JsonSubTypes.Type(value = RotateImageTransform.class, name = "RotateImageTransform"),
                @JsonSubTypes.Type(value = ScaleImageTransform.class, name = "ScaleImageTransform"),
                @JsonSubTypes.Type(value = WarpImageTransform.class, name = "WarpImageTransform")})

public interface ImageTransform {

    /**
     * Takes an image and returns a transformed image.
     *
     * @param image to transform, null == end of stream
     * @return      transformed image
     */
    ImageWritable transform(ImageWritable image);

    /**
     * Takes an image and returns a transformed image.
     * Uses the random object in the case of random transformations.
     *
     * @param image  to transform, null == end of stream
     * @param random object to use (or null for deterministic)
     * @return       transformed image
     */
    ImageWritable transform(ImageWritable image, Random random);

    /**
     * Transforms the given coordinates using the parameters that were used to transform the last image.
     *
     * @param coordinates to transforms (x1, y1, x2, y2, ...)
     * @return            transformed coordinates
     */
    float[] query(float... coordinates);

    /**
     * Returns the last transformed image or null if none transformed yet.
     * 
     * @return Last transformed image or null
     */
    ImageWritable getCurrentImage();
}
