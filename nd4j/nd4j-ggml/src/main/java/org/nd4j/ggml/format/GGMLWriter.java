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

package org.nd4j.ggml.format;

import lombok.extern.slf4j.Slf4j;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Writer for legacy GGML format files.
 * This is the reverse operation of {@link GGMLReader}.
 *
 * <p>Note: GGUF format is preferred for new models. This writer is provided
 * for compatibility with older tools that only support legacy GGML format.
 *
 * <p>Legacy GGML format structure:
 * - Magic number (4 bytes): 0x67676d6c ("ggml")
 * - Version (4 bytes)
 * - Model parameters (vocab_size, hidden_size, etc.)
 * - Tensor data
 */
@Slf4j
public class GGMLWriter implements Closeable {

    private final File outputFile;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final GGMLFormat format;

    private final List<TensorRegistration> tensors = new ArrayList<>();
    private boolean headerWritten = false;

    // Model parameters for legacy format
    private int vocabSize;
    private int hiddenSize;
    private int numLayers;
    private int numHeads;
    private int numKVHeads;
    private int contextLength;

    private static class TensorRegistration {
        String name;
        long[] shape;
        GGMLDataType dataType;
        byte[] data;
    }

    /**
     * Create a new GGML writer
     *
     * @param outputFile the output file
     * @param format     the GGML format (GGML, GGMF, or GGJT)
     */
    public GGMLWriter(File outputFile, GGMLFormat format) throws IOException {
        if (format == GGMLFormat.GGUF) {
            throw new IllegalArgumentException("Use GGUFWriter for GGUF format");
        }

        this.outputFile = outputFile;
        this.format = format;
        this.raf = new RandomAccessFile(outputFile, "rw");
        this.channel = raf.getChannel();

        // Truncate file if it exists
        channel.truncate(0);
    }

    /**
     * Set model parameters for the header
     */
    public void setModelParameters(int vocabSize, int hiddenSize, int numLayers,
                                    int numHeads, int numKVHeads, int contextLength) {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.numKVHeads = numKVHeads;
        this.contextLength = contextLength;
    }

    /**
     * Add a tensor to be written
     */
    public void addTensor(String name, long[] shape, GGMLDataType dataType, byte[] data) {
        if (headerWritten) {
            throw new IllegalStateException("Cannot add tensors after write() is called");
        }

        TensorRegistration reg = new TensorRegistration();
        reg.name = name;
        reg.shape = shape;
        reg.dataType = dataType;
        reg.data = data;
        tensors.add(reg);
    }

    /**
     * Write the complete GGML file
     */
    public void write() throws IOException {
        ByteBuffer buffer = ByteBuffer.allocate(1024 * 1024).order(ByteOrder.LITTLE_ENDIAN);

        // Write magic
        int magic = getMagicForFormat(format);
        buffer.putInt(magic);

        // Write format-specific header
        switch (format) {
            case GGML:
                writeGGMLHeader(buffer);
                break;
            case GGMF:
                writeGGMFHeader(buffer);
                break;
            case GGJT:
                writeGGJTHeader(buffer);
                break;
            default:
                throw new IOException("Unsupported format: " + format);
        }

        // Write to file
        buffer.flip();
        channel.write(buffer);

        // Write tensors
        for (TensorRegistration tensor : tensors) {
            writeTensor(tensor);
        }

        headerWritten = true;
        log.info("Wrote legacy GGML file: {} ({} tensors)", outputFile.getName(), tensors.size());
    }

    private void writeGGMLHeader(ByteBuffer buffer) {
        // Basic GGML header: just model parameters
        buffer.putInt(vocabSize);
        buffer.putInt(hiddenSize);
        buffer.putInt(numLayers);
        buffer.putInt(numHeads);
        buffer.putInt(contextLength);
    }

    private void writeGGMFHeader(ByteBuffer buffer) {
        // GGMF header includes version
        buffer.putInt(1); // Version
        buffer.putInt(vocabSize);
        buffer.putInt(hiddenSize);
        buffer.putInt(numLayers);
        buffer.putInt(numHeads);
        buffer.putInt(numKVHeads);
        buffer.putInt(contextLength);
    }

    private void writeGGJTHeader(ByteBuffer buffer) {
        // GGJT header includes version and more parameters
        buffer.putInt(3); // Version
        buffer.putInt(vocabSize);
        buffer.putInt(hiddenSize);
        buffer.putInt(numLayers);
        buffer.putInt(numHeads);
        buffer.putInt(numKVHeads);
        buffer.putInt(contextLength);
        buffer.putInt(0); // Padding/reserved
    }

    private void writeTensor(TensorRegistration tensor) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(256).order(ByteOrder.LITTLE_ENDIAN);

        // Number of dimensions
        header.putInt(tensor.shape.length);

        // Name length and name
        byte[] nameBytes = tensor.name.getBytes(StandardCharsets.UTF_8);
        header.putInt(nameBytes.length);
        header.put(nameBytes);

        // Shape
        for (long dim : tensor.shape) {
            header.putInt((int) dim); // Legacy format uses 32-bit for dimensions
        }

        // Type
        header.putInt(tensor.dataType.getTypeId());

        // Write header
        header.flip();
        channel.write(header);

        // Write data
        ByteBuffer dataBuffer = ByteBuffer.wrap(tensor.data);
        channel.write(dataBuffer);
    }

    private int getMagicForFormat(GGMLFormat format) {
        switch (format) {
            case GGML:
                return GGMLFormatDetector.GGML_MAGIC_GGML;
            case GGMF:
                return GGMLFormatDetector.GGML_MAGIC_GGMF;
            case GGJT:
                return GGMLFormatDetector.GGML_MAGIC_GGJT;
            default:
                throw new IllegalArgumentException("Unsupported format: " + format);
        }
    }

    @Override
    public void close() throws IOException {
        if (channel != null && channel.isOpen()) {
            channel.close();
        }
        if (raf != null) {
            raf.close();
        }
    }
}
