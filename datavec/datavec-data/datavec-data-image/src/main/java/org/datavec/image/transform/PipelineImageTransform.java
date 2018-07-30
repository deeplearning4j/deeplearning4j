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
import lombok.NonNull;

import org.datavec.image.data.ImageWritable;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

import static org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Allows creation of image transform pipelines, either sequentially or randomly.
 *
 * You have the option of passing in multiple transforms as parameters. If you
 * want to create a more complex pipeline, you can pass in a pipeline that looks
 * like {@literal List<Pair<ImageTransform, Double>>}. The Double value is the probability that
 * particular element in the pipeline will be executed. This is helpful for creating
 * complex pipelines.
 *
 * The pipeline can also be randomly shuffled with each transform, further increasing
 * the available dataset.
 *
 * @author saudet
 * @author crockpotveggies
 */
@Data
public class PipelineImageTransform extends BaseImageTransform<Mat> {

    protected List<Pair<ImageTransform, Double>> imageTransforms;
    protected boolean shuffle;
    protected org.nd4j.linalg.api.rng.Random rng;

    protected List<ImageTransform> currentTransforms = new ArrayList<>();

    public PipelineImageTransform(ImageTransform... transforms) {
        this(1234, false, transforms);
    }

    public PipelineImageTransform(long seed, boolean shuffle, ImageTransform... transforms) {
        super(null); // for perf reasons we ignore java Random, create our own

        List<Pair<ImageTransform, Double>> pipeline = new LinkedList<>();
        for (int i = 0; i < transforms.length; i++) {
            pipeline.add(new Pair<>(transforms[i], 1.0));
        }

        this.imageTransforms = pipeline;
        this.shuffle = shuffle;
        this.rng = Nd4j.getRandom();
        rng.setSeed(seed);
    }

    public PipelineImageTransform(List<Pair<ImageTransform, Double>> transforms) {
        this(1234, transforms, false);
    }

    public PipelineImageTransform(List<Pair<ImageTransform, Double>> transforms, boolean shuffle) {
        this(1234, transforms, shuffle);
    }

    public PipelineImageTransform(long seed, List<Pair<ImageTransform, Double>> transforms) {
        this(seed, transforms, false);
    }

    public PipelineImageTransform(long seed, List<Pair<ImageTransform, Double>> transforms, boolean shuffle) {
        this(null, seed, transforms, shuffle);
    }

    public PipelineImageTransform(Random random, long seed, List<Pair<ImageTransform, Double>> transforms,
            boolean shuffle) {
        super(random); // used by the transforms in the pipeline
        this.imageTransforms = transforms;
        this.shuffle = shuffle;
        this.rng = Nd4j.getRandom();
        rng.setSeed(seed);
    }

    /**
     * Takes an image and executes a pipeline of combined transforms.
     *
     * @param image to transform, null == end of stream
     * @param random object to use (or null for deterministic)
     * @return transformed image
     */
    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (shuffle) {
            Collections.shuffle(imageTransforms);
        }

        currentTransforms.clear();

        // execute each item in the pipeline
        for (Pair<ImageTransform, Double> tuple : imageTransforms) {
            if (tuple.getSecond() == 1.0 || rng.nextDouble() < tuple.getSecond()) { // probability of execution
                currentTransforms.add(tuple.getFirst());
                image = random != null ? tuple.getFirst().transform(image, random)
                        : tuple.getFirst().transform(image);
            }
        }

        return image;
    }

    @Override
    public float[] query(float... coordinates) {
        for (ImageTransform transform : currentTransforms) {
            coordinates = transform.query(coordinates);
        }
        return coordinates;
    }

    /**
     * Optional builder helper for PipelineImageTransform
     */
    public static class Builder {

        protected List<Pair<ImageTransform, Double>> imageTransforms = new ArrayList<>();
        protected Long seed = null;

        /**
         * This method sets RNG seet for this pipeline
         *
         * @param seed
         * @return
         */
        public Builder setSeed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * This method adds given transform with 100% invocation probability to this pipelien
         *
         * @param transform
         * @return
         */
        public Builder addImageTransform(@NonNull ImageTransform transform) {
            return addImageTransform(transform, 1.0);
        }

        /**
         * This method adds given transform with given invocation probability to this pipelien
         *
         * @param transform
         * @param probability
         * @return
         */
        public Builder addImageTransform(@NonNull ImageTransform transform, Double probability) {
            if (probability < 0.0) {
                probability = 0.0;
            }
            if (probability > 1.0) {
                probability = 1.0;
            }

            imageTransforms.add(Pair.makePair(transform, probability));
            return this;
        }

        /**
         * This method returns new PipelineImageTransform instance
         *
         * @return
         */
        public PipelineImageTransform build() {
            if (seed != null) {
                return new PipelineImageTransform(seed, imageTransforms);
            } else {
                return new PipelineImageTransform(imageTransforms);
            }
        }
    }
}
