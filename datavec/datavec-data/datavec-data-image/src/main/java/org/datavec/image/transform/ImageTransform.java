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
import org.datavec.image.data.ImageWritable;
import org.datavec.image.serde.LegacyImageMappingHelper;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.util.Random;

/**
 * Transforms an image in some way, either deterministically or randomly.
 *
 * @author saudet
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyImageMappingHelper.ImageTransformHelper.class)
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
