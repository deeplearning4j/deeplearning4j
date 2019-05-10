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

import java.io.FileOutputStream;
import java.io.IOException;


public class WaveFileManager {

    private Wave wave;

    public WaveFileManager() {
        wave = new Wave();
    }

    public WaveFileManager(Wave wave) {
        setWave(wave);
    }

    /**
     * Save the wave file
     * 
     * @param filename
     *            filename to be saved
     *            
     * @see	Wave file saved
     */
    public void saveWaveAsFile(String filename) {

        WaveHeader waveHeader = wave.getWaveHeader();

        int byteRate = waveHeader.getByteRate();
        int audioFormat = waveHeader.getAudioFormat();
        int sampleRate = waveHeader.getSampleRate();
        int bitsPerSample = waveHeader.getBitsPerSample();
        int channels = waveHeader.getChannels();
        long chunkSize = waveHeader.getChunkSize();
        long subChunk1Size = waveHeader.getSubChunk1Size();
        long subChunk2Size = waveHeader.getSubChunk2Size();
        int blockAlign = waveHeader.getBlockAlign();

        try {
            FileOutputStream fos = new FileOutputStream(filename);
            fos.write(WaveHeader.RIFF_HEADER.getBytes());
            // little endian
            fos.write(new byte[] {(byte) (chunkSize), (byte) (chunkSize >> 8), (byte) (chunkSize >> 16),
                            (byte) (chunkSize >> 24)});
            fos.write(WaveHeader.WAVE_HEADER.getBytes());
            fos.write(WaveHeader.FMT_HEADER.getBytes());
            fos.write(new byte[] {(byte) (subChunk1Size), (byte) (subChunk1Size >> 8), (byte) (subChunk1Size >> 16),
                            (byte) (subChunk1Size >> 24)});
            fos.write(new byte[] {(byte) (audioFormat), (byte) (audioFormat >> 8)});
            fos.write(new byte[] {(byte) (channels), (byte) (channels >> 8)});
            fos.write(new byte[] {(byte) (sampleRate), (byte) (sampleRate >> 8), (byte) (sampleRate >> 16),
                            (byte) (sampleRate >> 24)});
            fos.write(new byte[] {(byte) (byteRate), (byte) (byteRate >> 8), (byte) (byteRate >> 16),
                            (byte) (byteRate >> 24)});
            fos.write(new byte[] {(byte) (blockAlign), (byte) (blockAlign >> 8)});
            fos.write(new byte[] {(byte) (bitsPerSample), (byte) (bitsPerSample >> 8)});
            fos.write(WaveHeader.DATA_HEADER.getBytes());
            fos.write(new byte[] {(byte) (subChunk2Size), (byte) (subChunk2Size >> 8), (byte) (subChunk2Size >> 16),
                            (byte) (subChunk2Size >> 24)});
            fos.write(wave.getBytes());
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Wave getWave() {
        return wave;
    }

    public void setWave(Wave wave) {
        this.wave = wave;
    }
}
