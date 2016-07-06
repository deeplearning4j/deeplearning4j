/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
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

package org.canova.image.recordreader;


import java.util.List;

/**
 * Parse the label of the image from the name of the file
 * rather than the parent directory
 *
 * @author Adam Gibson
 */
public class ImageNameRecordReader extends BaseImageRecordReader {
    public ImageNameRecordReader() {
    }

    public ImageNameRecordReader(int height, int width, int channels, List<String> labels) {
        super(height, width, channels, labels);
    }

    public ImageNameRecordReader(int height, int width, int channels, boolean appendLabel, List<String> labels) {
        super(height, width, channels, appendLabel, labels);
    }

    public ImageNameRecordReader(int height, int width, int channels) {
        super(height, width, channels, false);
    }

    public ImageNameRecordReader(int height, int width, int channels, boolean appendLabel) {
        super(height, width, channels, appendLabel);
    }

    public ImageNameRecordReader(int height, int width, List<String> labels) {
        super(height, width, 1, labels);
    }

    public ImageNameRecordReader(int height, int width, boolean appendLabel, List<String> labels) {
        super(height, width, 1, appendLabel, labels);
    }

    public ImageNameRecordReader(int height, int width) {
        super(height, width, 1, false);
    }

    public ImageNameRecordReader(int height, int width, boolean appendLabel) {
        super(height, width, 1, appendLabel);
    }

    @Override
    public String getLabel(String path) {
        int startOfFormat = path.lastIndexOf('.');
        if(startOfFormat < 0)
            throw new IllegalStateException("Illegal path; no format found");
        StringBuilder label = new StringBuilder();
        while(path.charAt(startOfFormat) != '-') {
            label.append(path.charAt(startOfFormat));
            startOfFormat--;
        }

        if(startOfFormat < 0)
            throw new IllegalStateException("Illegal path; no - found. A dash is used to inidicate a label.");
        return label.reverse().toString();
    }

}
