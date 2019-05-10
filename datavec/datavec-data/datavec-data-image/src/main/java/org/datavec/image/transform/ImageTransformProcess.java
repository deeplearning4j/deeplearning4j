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
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.transform.serde.JsonMappers;
import org.datavec.api.writable.Writable;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.core.JsonProcessingException;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by kepricon on 17. 5. 23.
 */
@Data
@Slf4j
@NoArgsConstructor
public class ImageTransformProcess {

    private List<ImageTransform> transformList;
    private int seed;

    public ImageTransformProcess(int seed, ImageTransform... transforms) {
        this.seed = seed;
        this.transformList = Arrays.asList(transforms);
    }

    public ImageTransformProcess(int seed, List<ImageTransform> transformList) {
        this.seed = seed;
        this.transformList = transformList;
    }

    public ImageTransformProcess(Builder builder) {
        this(builder.seed, builder.transformList);
    }

    public List<Writable> execute(List<Writable> image) {
        throw new UnsupportedOperationException();
    }

    public INDArray executeArray(ImageWritable image) throws IOException {
        Random random = null;
        if (seed != 0) {
            random = new Random(seed);
        }

        ImageWritable currentImage = image;
        for (ImageTransform transform : transformList) {
            currentImage = transform.transform(currentImage, random);
        }

        NativeImageLoader imageLoader = new NativeImageLoader();
        return imageLoader.asMatrix(currentImage);
    }

    public ImageWritable execute(ImageWritable image) throws IOException {
        Random random = null;
        if (seed != 0) {
            random = new Random(seed);
        }

        ImageWritable currentImage = image;
        for (ImageTransform transform : transformList) {
            currentImage = transform.transform(currentImage, random);
        }

        return currentImage;
    }

    public ImageWritable transformFileUriToInput(URI uri) throws IOException {

        NativeImageLoader imageLoader = new NativeImageLoader();
        ImageWritable img = imageLoader.asWritable(new File(uri));

        return img;
    }

    /**
     * Convert the ImageTransformProcess to a JSON string
     *
     * @return ImageTransformProcess, as JSON
     */
    public String toJson() {
        try {
            return JsonMappers.getMapper().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            //TODO better exceptions
            throw new RuntimeException(e);
        }
    }

    /**
     * Convert the ImageTransformProcess to a YAML string
     *
     * @return ImageTransformProcess, as YAML
     */
    public String toYaml() {
        try {
            return JsonMappers.getMapperYaml().writeValueAsString(this);
        } catch (JsonProcessingException e) {
            //TODO better exceptions
            throw new RuntimeException(e);
        }
    }

    /**
     * Deserialize a JSON String (created by {@link #toJson()}) to a ImageTransformProcess
     *
     * @return ImageTransformProcess, from JSON
     */
    public static ImageTransformProcess fromJson(String json) {
        try {
            return JsonMappers.getMapper().readValue(json, ImageTransformProcess.class);
        } catch (IOException e) {
            //TODO better exceptions
            throw new RuntimeException(e);
        }
    }

    /**
     * Deserialize a JSON String (created by {@link #toJson()}) to a ImageTransformProcess
     *
     * @return ImageTransformProcess, from JSON
     */
    public static ImageTransformProcess fromYaml(String yaml) {
        try {
            return JsonMappers.getMapperYaml().readValue(yaml, ImageTransformProcess.class);
        } catch (IOException e) {
            //TODO better exceptions
            throw new RuntimeException(e);
        }
    }

    /**
     * Builder class for constructing a ImageTransformProcess
     */
    public static class Builder {

        private List<ImageTransform> transformList;
        private int seed = 0;

        public Builder() {
            transformList = new ArrayList<>();
        }

        public Builder seed(int seed) {
            this.seed = seed;
            return this;
        }

        public Builder cropImageTransform(int crop) {
            transformList.add(new CropImageTransform(crop));
            return this;
        }

        public Builder cropImageTransform(int cropTop, int cropLeft, int cropBottom, int cropRight) {
            transformList.add(new CropImageTransform(cropTop, cropLeft, cropBottom, cropRight));
            return this;
        }

        public Builder colorConversionTransform(int conversionCode) {
            transformList.add(new ColorConversionTransform(conversionCode));
            return this;
        }

        public Builder equalizeHistTransform(int conversionCode) {
            transformList.add(new EqualizeHistTransform(conversionCode));
            return this;
        }

        public Builder filterImageTransform(String filters, int width, int height) {
            transformList.add(new FilterImageTransform(filters, width, height));
            return this;
        }

        public Builder filterImageTransform(String filters, int width, int height, int channels) {
            transformList.add(new FilterImageTransform(filters, width, height, channels));
            return this;
        }

        public Builder flipImageTransform(int flipMode) {
            transformList.add(new FlipImageTransform(flipMode));
            return this;
        }

        public Builder randomCropTransform(int height, int width) {
            transformList.add(new RandomCropTransform(height, width));
            return this;
        }

        public Builder randomCropTransform(long seed, int height, int width) {
            transformList.add(new RandomCropTransform(seed, height, width));
            return this;
        }

        public Builder resizeImageTransform(int newWidth, int newHeight) {
            transformList.add(new ResizeImageTransform(newWidth, newHeight));
            return this;
        }

        public Builder rotateImageTransform(float angle) {
            transformList.add(new RotateImageTransform(angle));
            return this;
        }

        public Builder rotateImageTransform(float centerx, float centery, float angle, float scale) {
            transformList.add(new RotateImageTransform(centerx, centery, angle, scale));
            return this;
        }

        public Builder scaleImageTransform(float delta) {
            transformList.add(new ScaleImageTransform(delta));
            return this;
        }

        public Builder scaleImageTransform(float dx, float dy) {
            transformList.add(new ScaleImageTransform(dx, dy));
            return this;
        }

        public Builder warpImageTransform(float delta) {
            transformList.add(new WarpImageTransform(delta));
            return this;
        }

        public Builder warpImageTransform(float dx1, float dy1, float dx2, float dy2, float dx3, float dy3, float dx4,
                        float dy4) {
            transformList.add(new WarpImageTransform(dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4));
            return this;
        }


        public ImageTransformProcess build() {
            return new ImageTransformProcess(this);
        }

    }
}
