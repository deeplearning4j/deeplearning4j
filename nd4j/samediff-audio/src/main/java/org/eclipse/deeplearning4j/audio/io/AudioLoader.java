/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.audio.io;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Utility for loading audio files (WAV format) into INDArrays.
 * Uses Java Sound API for audio decoding.
 */
@Slf4j
public class AudioLoader {

    /**
     * Load a WAV file into an INDArray.
     *
     * @param file Path to the WAV file
     * @return Audio samples as a FLOAT INDArray of shape [samples]
     * @throws IOException if the file cannot be read
     */
    public static INDArray loadWav(File file) throws IOException {
        try (AudioInputStream audioStream = AudioSystem.getAudioInputStream(file)) {
            return readAudioStream(audioStream);
        } catch (Exception e) {
            throw new IOException("Failed to load audio file: " + file.getAbsolutePath(), e);
        }
    }

    /**
     * Load a WAV file into an INDArray.
     *
     * @param path Path to the WAV file
     * @return Audio samples as a FLOAT INDArray of shape [samples]
     * @throws IOException if the file cannot be read
     */
    public static INDArray loadWav(String path) throws IOException {
        return loadWav(new File(path));
    }

    /**
     * Get the sample rate of an audio file.
     *
     * @param file Path to the audio file
     * @return Sample rate in Hz
     * @throws IOException if the file cannot be read
     */
    public static int getSampleRate(File file) throws IOException {
        try (AudioInputStream audioStream = AudioSystem.getAudioInputStream(file)) {
            return (int) audioStream.getFormat().getSampleRate();
        } catch (Exception e) {
            throw new IOException("Failed to read audio file: " + file.getAbsolutePath(), e);
        }
    }

    private static INDArray readAudioStream(AudioInputStream audioStream) throws IOException {
        AudioFormat format = audioStream.getFormat();

        // Convert to PCM if needed
        if (format.getEncoding() != AudioFormat.Encoding.PCM_SIGNED &&
            format.getEncoding() != AudioFormat.Encoding.PCM_FLOAT) {
            AudioFormat targetFormat = new AudioFormat(
                    AudioFormat.Encoding.PCM_SIGNED,
                    format.getSampleRate(),
                    16,
                    format.getChannels(),
                    format.getChannels() * 2,
                    format.getSampleRate(),
                    false
            );
            audioStream = AudioSystem.getAudioInputStream(targetFormat, audioStream);
            format = targetFormat;
        }

        byte[] audioBytes = audioStream.readAllBytes();
        int bytesPerSample = format.getSampleSizeInBits() / 8;
        int numChannels = format.getChannels();
        int numSamples = audioBytes.length / (bytesPerSample * numChannels);

        float[] samples = new float[numSamples];
        ByteBuffer buffer = ByteBuffer.wrap(audioBytes).order(
                format.isBigEndian() ? ByteOrder.BIG_ENDIAN : ByteOrder.LITTLE_ENDIAN);

        for (int i = 0; i < numSamples; i++) {
            float sample = 0;

            // Read and average all channels (mono mixdown)
            for (int ch = 0; ch < numChannels; ch++) {
                float channelSample;
                switch (bytesPerSample) {
                    case 1:
                        channelSample = (buffer.get() & 0xFF - 128) / 128.0f;
                        break;
                    case 2:
                        channelSample = buffer.getShort() / 32768.0f;
                        break;
                    case 3:
                        int val = buffer.get() & 0xFF;
                        val |= (buffer.get() & 0xFF) << 8;
                        val |= buffer.get() << 16;
                        channelSample = val / 8388608.0f;
                        break;
                    case 4:
                        if (format.getEncoding() == AudioFormat.Encoding.PCM_FLOAT) {
                            channelSample = buffer.getFloat();
                        } else {
                            channelSample = buffer.getInt() / 2147483648.0f;
                        }
                        break;
                    default:
                        throw new IOException("Unsupported sample size: " + bytesPerSample + " bytes");
                }
                sample += channelSample;
            }

            samples[i] = sample / numChannels;
        }

        return Nd4j.createFromArray(samples);
    }
}
