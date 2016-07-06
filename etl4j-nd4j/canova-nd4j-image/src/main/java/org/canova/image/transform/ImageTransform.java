/*
 *
 *  *
 *  *  * Copyright 2016 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */
package org.canova.image.transform;

import java.util.Random;
import org.canova.image.data.ImageWritable;

/**
 * Transforms an image in some way, either deterministically or randomly.
 *
 * @author saudet
 */
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

}
