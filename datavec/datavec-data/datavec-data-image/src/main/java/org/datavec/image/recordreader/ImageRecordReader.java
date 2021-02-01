/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.image.recordreader;


import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.PathMultiLabelGenerator;
import org.datavec.image.transform.ImageTransform;

/**
 * Image record reader.
 * Reads a local file system and parses images of a given
 * height and width.
 * All images are rescaled and converted to the given height, width, and number of channels.
 *
 * Also appends the label if specified
 * (one of k encoding based on the directory structure where each subdir of the root is an indexed label)
 * @author Adam Gibson
 */
public class ImageRecordReader extends BaseImageRecordReader {


    /** Loads images with height = 28, width = 28, and channels = 1, appending no labels.
     * Output format is NCHW (channels first) - [numExamples, 1, 28, 28]*/
    public ImageRecordReader() {
        super();
    }

    /** Loads images with given height, width, and channels, appending labels returned by the generator.
     * Output format is NCHW (channels first) - [numExamples, channels, height, width]
     */
    public ImageRecordReader(long height, long width, long channels, PathLabelGenerator labelGenerator) {
        super(height, width, channels, labelGenerator);
    }

    /** Loads images with given height, width, and channels, appending labels returned by the generator.
     * Output format is NCHW (channels first) - [numExamples, channels, height, width]
     */
    public ImageRecordReader(long height, long width, long channels, PathMultiLabelGenerator labelGenerator) {
        super(height, width, channels, labelGenerator);
    }

    /** Loads images with given height, width, and channels, appending no labels - in NCHW (channels first) format */
    public ImageRecordReader(long height, long width, long channels) {
        super(height, width, channels, (PathLabelGenerator) null);
    }

    /** Loads images with given height, width, and channels, appending no labels - in specified format<br>
     * If {@code nchw_channels_first == true} output format is NCHW (channels first) - [numExamples, channels, height, width]<br>
     * If {@code nchw_channels_first == false} output format is NHWC (channels last) - [numExamples, height, width, channels]<br>
     */
    public ImageRecordReader(long height, long width, long channels, boolean nchw_channels_first) {
        super(height, width, channels, nchw_channels_first, null, null,  null);
    }

    /** Loads images with given height, width, and channels, appending labels returned by the generator.
     * Output format is NCHW (channels first) - [numExamples, channels, height, width] */
    public ImageRecordReader(long height, long width, long channels, PathLabelGenerator labelGenerator,
                    ImageTransform imageTransform) {
        super(height, width, channels, labelGenerator, imageTransform);
    }

    /** Loads images with given height, width, and channels, appending labels returned by the generator.<br>
     * If {@code nchw_channels_first == true} output format is NCHW (channels first) - [numExamples, channels, height, width]<br>
     * If {@code nchw_channels_first == false} output format is NHWC (channels last) - [numExamples, height, width, channels]<br>
     */
    public ImageRecordReader(long height, long width, long channels, boolean nchw_channels_first, PathLabelGenerator labelGenerator,
                             ImageTransform imageTransform) {
        super(height, width, channels, nchw_channels_first, labelGenerator, null, imageTransform);
    }

    /** Loads images with given height, width, and channels, appending no labels.
     * Output format is NCHW (channels first) - [numExamples, channels, height, width]*/
    public ImageRecordReader(long height, long width, long channels, ImageTransform imageTransform) {
        super(height, width, channels, null, imageTransform);
    }

    /** Loads images with given  height, width, and channels, appending labels returned by the generator
     * Output format is NCHW (channels first) - [numExamples, channels, height, width]*/
    public ImageRecordReader(long height, long width, PathLabelGenerator labelGenerator) {
        super(height, width, 1, labelGenerator);
    }

    /** Loads images with given height, width, and channels = 1, appending no labels.
     * Output format is NCHW (channels first) - [numExamples, channels, height, width]*/
    public ImageRecordReader(long height, long width) {
        super(height, width, 1, null, null);
    }
}
