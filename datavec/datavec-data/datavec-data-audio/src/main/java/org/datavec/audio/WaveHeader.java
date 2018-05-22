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

package org.datavec.audio;

import java.io.IOException;
import java.io.InputStream;

/**
 * WAV File Specification
 * https://ccrma.stanford.edu/courses/422/projects/WaveFormat/
 * 
 * @author Jacquet Wong
 */
public class WaveHeader {

    public static final String RIFF_HEADER = "RIFF";
    public static final String WAVE_HEADER = "WAVE";
    public static final String FMT_HEADER = "fmt ";
    public static final String DATA_HEADER = "data";
    public static final int HEADER_BYTE_LENGTH = 44; // 44 bytes for header

    private boolean valid;
    private String chunkId; // 4 bytes
    private long chunkSize; // unsigned 4 bytes, little endian
    private String format; // 4 bytes
    private String subChunk1Id; // 4 bytes
    private long subChunk1Size; // unsigned 4 bytes, little endian
    private int audioFormat; // unsigned 2 bytes, little endian
    private int channels; // unsigned 2 bytes, little endian
    private long sampleRate; // unsigned 4 bytes, little endian
    private long byteRate; // unsigned 4 bytes, little endian
    private int blockAlign; // unsigned 2 bytes, little endian
    private int bitsPerSample; // unsigned 2 bytes, little endian
    private String subChunk2Id; // 4 bytes
    private long subChunk2Size; // unsigned 4 bytes, little endian

    public WaveHeader() {
        // init a 8k 16bit mono wav		
        chunkSize = 36;
        subChunk1Size = 16;
        audioFormat = 1;
        channels = 1;
        sampleRate = 8000;
        byteRate = 16000;
        blockAlign = 2;
        bitsPerSample = 16;
        subChunk2Size = 0;
        valid = true;
    }

    public WaveHeader(InputStream inputStream) {
        valid = loadHeader(inputStream);
    }

    private boolean loadHeader(InputStream inputStream) {

        byte[] headerBuffer = new byte[HEADER_BYTE_LENGTH];
        try {
            inputStream.read(headerBuffer);

            // read header
            int pointer = 0;
            chunkId = new String(new byte[] {headerBuffer[pointer++], headerBuffer[pointer++], headerBuffer[pointer++],
                            headerBuffer[pointer++]});
            // little endian
            chunkSize = (long) (headerBuffer[pointer++] & 0xff) | (long) (headerBuffer[pointer++] & 0xff) << 8
                            | (long) (headerBuffer[pointer++] & 0xff) << 16
                            | (long) (headerBuffer[pointer++] & 0xff << 24);
            format = new String(new byte[] {headerBuffer[pointer++], headerBuffer[pointer++], headerBuffer[pointer++],
                            headerBuffer[pointer++]});
            subChunk1Id = new String(new byte[] {headerBuffer[pointer++], headerBuffer[pointer++],
                            headerBuffer[pointer++], headerBuffer[pointer++]});
            subChunk1Size = (long) (headerBuffer[pointer++] & 0xff) | (long) (headerBuffer[pointer++] & 0xff) << 8
                            | (long) (headerBuffer[pointer++] & 0xff) << 16
                            | (long) (headerBuffer[pointer++] & 0xff) << 24;
            audioFormat = (int) ((headerBuffer[pointer++] & 0xff) | (headerBuffer[pointer++] & 0xff) << 8);
            channels = (int) ((headerBuffer[pointer++] & 0xff) | (headerBuffer[pointer++] & 0xff) << 8);
            sampleRate = (long) (headerBuffer[pointer++] & 0xff) | (long) (headerBuffer[pointer++] & 0xff) << 8
                            | (long) (headerBuffer[pointer++] & 0xff) << 16
                            | (long) (headerBuffer[pointer++] & 0xff) << 24;
            byteRate = (long) (headerBuffer[pointer++] & 0xff) | (long) (headerBuffer[pointer++] & 0xff) << 8
                            | (long) (headerBuffer[pointer++] & 0xff) << 16
                            | (long) (headerBuffer[pointer++] & 0xff) << 24;
            blockAlign = (int) ((headerBuffer[pointer++] & 0xff) | (headerBuffer[pointer++] & 0xff) << 8);
            bitsPerSample = (int) ((headerBuffer[pointer++] & 0xff) | (headerBuffer[pointer++] & 0xff) << 8);
            subChunk2Id = new String(new byte[] {headerBuffer[pointer++], headerBuffer[pointer++],
                            headerBuffer[pointer++], headerBuffer[pointer++]});
            subChunk2Size = (long) (headerBuffer[pointer++] & 0xff) | (long) (headerBuffer[pointer++] & 0xff) << 8
                            | (long) (headerBuffer[pointer++] & 0xff) << 16
                            | (long) (headerBuffer[pointer++] & 0xff) << 24;
            // end read header

            // the inputStream should be closed outside this method

            // dis.close();

        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }

        if (bitsPerSample != 8 && bitsPerSample != 16) {
            System.err.println("WaveHeader: only supports bitsPerSample 8 or 16");
            return false;
        }

        // check the format is support
        if (chunkId.toUpperCase().equals(RIFF_HEADER) && format.toUpperCase().equals(WAVE_HEADER) && audioFormat == 1) {
            return true;
        } else {
            System.err.println("WaveHeader: Unsupported header format");
        }

        return false;
    }

    public boolean isValid() {
        return valid;
    }

    public String getChunkId() {
        return chunkId;
    }

    public long getChunkSize() {
        return chunkSize;
    }

    public String getFormat() {
        return format;
    }

    public String getSubChunk1Id() {
        return subChunk1Id;
    }

    public long getSubChunk1Size() {
        return subChunk1Size;
    }

    public int getAudioFormat() {
        return audioFormat;
    }

    public int getChannels() {
        return channels;
    }

    public int getSampleRate() {
        return (int) sampleRate;
    }

    public int getByteRate() {
        return (int) byteRate;
    }

    public int getBlockAlign() {
        return blockAlign;
    }

    public int getBitsPerSample() {
        return bitsPerSample;
    }

    public String getSubChunk2Id() {
        return subChunk2Id;
    }

    public long getSubChunk2Size() {
        return subChunk2Size;
    }

    public void setSampleRate(int sampleRate) {
        int newSubChunk2Size = (int) (this.subChunk2Size * sampleRate / this.sampleRate);
        // if num bytes for each sample is even, the size of newSubChunk2Size also needed to be in even number
        if ((bitsPerSample / 8) % 2 == 0) {
            if (newSubChunk2Size % 2 != 0) {
                newSubChunk2Size++;
            }
        }

        this.sampleRate = sampleRate;
        this.byteRate = sampleRate * bitsPerSample / 8;
        this.chunkSize = newSubChunk2Size + 36;
        this.subChunk2Size = newSubChunk2Size;
    }

    public void setChunkId(String chunkId) {
        this.chunkId = chunkId;
    }

    public void setChunkSize(long chunkSize) {
        this.chunkSize = chunkSize;
    }

    public void setFormat(String format) {
        this.format = format;
    }

    public void setSubChunk1Id(String subChunk1Id) {
        this.subChunk1Id = subChunk1Id;
    }

    public void setSubChunk1Size(long subChunk1Size) {
        this.subChunk1Size = subChunk1Size;
    }

    public void setAudioFormat(int audioFormat) {
        this.audioFormat = audioFormat;
    }

    public void setChannels(int channels) {
        this.channels = channels;
    }

    public void setByteRate(long byteRate) {
        this.byteRate = byteRate;
    }

    public void setBlockAlign(int blockAlign) {
        this.blockAlign = blockAlign;
    }

    public void setBitsPerSample(int bitsPerSample) {
        this.bitsPerSample = bitsPerSample;
    }

    public void setSubChunk2Id(String subChunk2Id) {
        this.subChunk2Id = subChunk2Id;
    }

    public void setSubChunk2Size(long subChunk2Size) {
        this.subChunk2Size = subChunk2Size;
    }

    public String toString() {

        StringBuilder sb = new StringBuilder();
        sb.append("chunkId: " + chunkId);
        sb.append("\n");
        sb.append("chunkSize: " + chunkSize);
        sb.append("\n");
        sb.append("format: " + format);
        sb.append("\n");
        sb.append("subChunk1Id: " + subChunk1Id);
        sb.append("\n");
        sb.append("subChunk1Size: " + subChunk1Size);
        sb.append("\n");
        sb.append("audioFormat: " + audioFormat);
        sb.append("\n");
        sb.append("channels: " + channels);
        sb.append("\n");
        sb.append("sampleRate: " + sampleRate);
        sb.append("\n");
        sb.append("byteRate: " + byteRate);
        sb.append("\n");
        sb.append("blockAlign: " + blockAlign);
        sb.append("\n");
        sb.append("bitsPerSample: " + bitsPerSample);
        sb.append("\n");
        sb.append("subChunk2Id: " + subChunk2Id);
        sb.append("\n");
        sb.append("subChunk2Size: " + subChunk2Size);
        return sb.toString();
    }
}
