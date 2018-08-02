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

import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Transforms images deterministically or randomly with the help of an array of ImageTransform
 *
 * @author saudet
 */
@Data
public class MultiImageTransform extends BaseImageTransform<Mat> {
    private PipelineImageTransform transform;

    public MultiImageTransform(ImageTransform... transforms) {
        this(null, transforms);
    }

    public MultiImageTransform(Random random, ImageTransform... transforms) {
        super(random);
        transform = new PipelineImageTransform(transforms);
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        return random == null ? transform.transform(image) : transform.transform(image, random);
    }

    @Override
    public float[] query(float... coordinates) {
        return transform.query(coordinates);
    }
}
