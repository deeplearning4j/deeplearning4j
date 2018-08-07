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
import org.bytedeco.javacv.FFmpegFrameFilter;
import org.bytedeco.javacv.FrameFilter;
import org.datavec.image.data.ImageWritable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Random;

import static org.bytedeco.javacpp.avutil.*;

/**
 * Filters images using FFmpeg (libavfilter):
 * <a href="https://ffmpeg.org/ffmpeg-filters.html">FFmpeg Filters Documentation</a>.
 *
 * @author saudet
 * @see FFmpegFrameFilter
 */
@JsonIgnoreProperties({"filter", "converter"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class FilterImageTransform extends BaseImageTransform {

    private FFmpegFrameFilter filter;

    private String filters;
    private int width;
    private int height;
    private int channels;

    /** Calls {@code this(filters, width, height, 3)}. */
    public FilterImageTransform(String filters, int width, int height) {
        this(filters, width, height, 3);
    }

    /**
     * Constructs a filtergraph out of the filter specification.
     *
     * @param filters  to use
     * @param width    of the input images
     * @param height   of the input images
     * @param channels of the input images
     */
    public FilterImageTransform(@JsonProperty("filters") String filters, @JsonProperty("width") int width,
                    @JsonProperty("height") int height, @JsonProperty("channels") int channels) {
        super(null);

        this.filters = filters;
        this.width = width;
        this.height = height;
        this.channels = channels;

        int pixelFormat = channels == 1 ? AV_PIX_FMT_GRAY8
                        : channels == 3 ? AV_PIX_FMT_BGR24 : channels == 4 ? AV_PIX_FMT_RGBA : AV_PIX_FMT_NONE;
        if (pixelFormat == AV_PIX_FMT_NONE) {
            throw new IllegalArgumentException("Unsupported number of channels: " + channels);
        }
        try {
            filter = new FFmpegFrameFilter(filters, width, height);
            filter.setPixelFormat(pixelFormat);
            filter.start();
        } catch (FrameFilter.Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected ImageWritable doTransform(ImageWritable image, Random random) {
        if (image == null) {
            return null;
        }
        try {
            filter.push(image.getFrame());
            image = new ImageWritable(filter.pull());
        } catch (FrameFilter.Exception e) {
            throw new RuntimeException(e);
        }
        return image;
    }

}
