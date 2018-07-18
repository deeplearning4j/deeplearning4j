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

package org.datavec.audio;

import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Frame;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.audio.recordreader.NativeAudioRecordReader;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.nio.ShortBuffer;
import java.util.List;

import static org.bytedeco.javacpp.avcodec.AV_CODEC_ID_VORBIS;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author saudet
 */
public class AudioReaderTest {
    @Ignore
    @Test
    public void testNativeAudioReader() throws Exception {
        File tempFile = File.createTempFile("testNativeAudioReader", ".ogg");
        FFmpegFrameRecorder recorder = new FFmpegFrameRecorder(tempFile, 2);
        recorder.setAudioCodec(AV_CODEC_ID_VORBIS);
        recorder.setSampleRate(44100);
        recorder.start();
        Frame audioFrame = new Frame();
        ShortBuffer audioBuffer = ShortBuffer.allocate(64 * 1024);
        audioFrame.sampleRate = 44100;
        audioFrame.audioChannels = 2;
        audioFrame.samples = new ShortBuffer[] {audioBuffer};
        recorder.record(audioFrame);
        recorder.stop();
        recorder.release();

        RecordReader reader = new NativeAudioRecordReader();
        reader.initialize(new FileSplit(tempFile));
        assertTrue(reader.hasNext());
        List<Writable> record = reader.next();
        assertEquals(audioBuffer.limit(), record.size());
    }
}
