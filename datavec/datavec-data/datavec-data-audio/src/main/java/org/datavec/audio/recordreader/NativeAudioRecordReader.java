/*-
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.audio.recordreader;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.avutil.AV_SAMPLE_FMT_FLT;

/**
 * Native audio file loader using FFmpeg.
 *
 * @author saudet
 */
public class NativeAudioRecordReader extends BaseAudioRecordReader {

    public NativeAudioRecordReader() {}

    public NativeAudioRecordReader(boolean appendLabel, List<String> labels) {
        super(appendLabel, labels);
    }

    public NativeAudioRecordReader(List<String> labels) {
        super(labels);
    }

    public NativeAudioRecordReader(boolean appendLabel) {
        super(appendLabel);
    }

    protected List<Writable> loadData(File file, InputStream inputStream) throws IOException {
        List<Writable> ret = new ArrayList<>();
        try (FFmpegFrameGrabber grabber = inputStream != null ? new FFmpegFrameGrabber(inputStream)
                        : new FFmpegFrameGrabber(file.getAbsolutePath())) {
            grabber.setSampleFormat(AV_SAMPLE_FMT_FLT);
            grabber.start();
            Frame frame;
            while ((frame = grabber.grab()) != null) {
                while (frame.samples != null && frame.samples[0].hasRemaining()) {
                    for (int i = 0; i < frame.samples.length; i++) {
                        ret.add(new FloatWritable(((FloatBuffer) frame.samples[i]).get()));
                    }
                }
            }
        }
        return ret;
    }

}
