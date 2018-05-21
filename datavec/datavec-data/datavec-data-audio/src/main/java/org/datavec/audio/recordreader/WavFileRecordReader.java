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

package org.datavec.audio.recordreader;

import org.datavec.api.util.RecordUtils;
import org.datavec.api.writable.Writable;
import org.datavec.audio.Wave;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

/**
 * Wav file loader
 * @author Adam Gibson
 */
public class WavFileRecordReader extends BaseAudioRecordReader {

    public WavFileRecordReader() {}

    public WavFileRecordReader(boolean appendLabel, List<String> labels) {
        super(appendLabel, labels);
    }

    public WavFileRecordReader(List<String> labels) {
        super(labels);
    }

    public WavFileRecordReader(boolean appendLabel) {
        super(appendLabel);
    }

    protected List<Writable> loadData(File file, InputStream inputStream) throws IOException {
        Wave wave = inputStream != null ? new Wave(inputStream) : new Wave(file.getAbsolutePath());
        return RecordUtils.toRecord(wave.getNormalizedAmplitudes());
    }

}
