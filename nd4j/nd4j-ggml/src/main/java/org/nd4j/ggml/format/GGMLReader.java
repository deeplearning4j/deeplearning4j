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
import org.nd4j.ggml.GGMLImportException;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Reader for legacy GGML format files (GGML, GGMF, GGJT).
 * These formats predate GGUF and have model-specific header layouts.
 */
@Slf4j
public class GGMLReader implements Closeable {

    private final File file;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private MappedByteBuffer buffer;

    private GGMLHeader header;
    private List<GGMLTensorInfo> tensorInfos;
    private long dataOffset;

    /**
     * Create a new GGML reader for the given file
     */
    public GGMLReader(File file) throws IOException {
        this.file = file;
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();

        this.buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, Math.min(channel.size(), Integer.MAX_VALUE));
        this.buffer.order(ByteOrder.LITTLE_ENDIAN);
    }

    /**
     * Read and parse the GGML header.
     * Note: The header structure varies by model type (LLaMA, GPT-2, etc.)
     * This implementation assumes LLaMA-style header layout.
     */
    public GGMLHeader readHeader() throws IOException, GGMLImportException {
        buffer.position(0);

        // Read magic
        int magic = buffer.getInt();
        GGMLHeader.LegacyFormat format = detectLegacyFormat(magic);
        if (format == GGMLHeader.LegacyFormat.UNKNOWN) {
            throw new GGMLImportException(String.format("Unknown GGML magic: 0x%08X", magic));
        }

        log.debug("Legacy GGML format: {}", format);

        // Version (for GGMF and GGJT)
        int version = 0;
        if (format != GGMLHeader.LegacyFormat.GGML) {
            version = buffer.getInt();
        }

        // Read hyperparameters (LLaMA-style layout)
        int vocabSize = buffer.getInt();
        int embeddingSize = buffer.getInt();
        int hiddenSize = buffer.getInt();
        int numLayers = buffer.getInt();
        int numHeads = buffer.getInt();
        int numKVHeads = buffer.getInt();
        int fileType = buffer.getInt();

        header = GGMLHeader.builder()
                .magic(magic)
                .version(version)
                .vocabSize(vocabSize)
                .embeddingSize(embeddingSize)
                .hiddenSize(hiddenSize)
                .numLayers(numLayers)
                .numHeads(numHeads)
                .numKVHeads(numKVHeads)
                .fileType(fileType)
                .build();

        log.debug("GGML header: vocab={}, embed={}, layers={}, heads={}",
                vocabSize, embeddingSize, numLayers, numHeads);

        return header;
    }

    /**
     * Read tensor information from the file
     */
    public List<GGMLTensorInfo> readTensorInfos() throws IOException, GGMLImportException {
        if (header == null) {
            readHeader();
        }

        tensorInfos = new ArrayList<>();

        // Skip vocabulary section (varies by model)
        skipVocabulary();

        // Read tensors until end of file or error
        long tensorDataStart = buffer.position();
        int tensorIndex = 0;

        try {
            while (buffer.hasRemaining()) {
                // Check if we have enough bytes for a tensor header
                if (buffer.remaining() < 12) { // minimum: n_dims(4) + name_len(4) + type(4)
                    break;
                }

                int startPos = buffer.position();

                int numDimensions = buffer.getInt();
                if (numDimensions < 0 || numDimensions > 4) {
                    // Likely reached end of tensor info section
                    buffer.position(startPos);
                    break;
                }

                int nameLength = buffer.getInt();
                if (nameLength < 0 || nameLength > 256) {
                    buffer.position(startPos);
                    break;
                }

                int typeId = buffer.getInt();
                if (!GGMLDataType.isValidTypeId(typeId)) {
                    buffer.position(startPos);
                    break;
                }

                // Read dimensions
                long[] shape = new long[numDimensions];
                for (int d = 0; d < numDimensions; d++) {
                    shape[d] = buffer.getInt() & 0xFFFFFFFFL;
                }

                // Read name
                byte[] nameBytes = new byte[nameLength];
                buffer.get(nameBytes);
                String name = new String(nameBytes, StandardCharsets.UTF_8);

                GGMLDataType dataType = GGMLDataType.fromTypeId(typeId);

                // Calculate data offset (align for GGJT)
                long currentPos = buffer.position();
                if (header.getLegacyFormat() == GGMLHeader.LegacyFormat.GGJT) {
                    currentPos = alignOffset(currentPos, 32);
                    buffer.position((int) currentPos);
                }

                GGMLTensorInfo tensorInfo = GGMLTensorInfo.builder()
                        .name(name)
                        .numDimensions(numDimensions)
                        .shape(shape)
                        .dataType(dataType)
                        .dataOffset(currentPos - tensorDataStart)
                        .build();

                tensorInfos.add(tensorInfo);

                // Skip tensor data
                long dataSize = tensorInfo.getDataSize();
                buffer.position((int) (buffer.position() + dataSize));

                if (log.isDebugEnabled()) {
                    log.debug("Tensor {}: {} {} type={}", tensorIndex, name,
                            tensorInfo.getShapeString(), dataType);
                }

                tensorIndex++;
            }
        } catch (Exception e) {
            log.warn("Error reading tensor {}: {}", tensorIndex, e.getMessage());
        }

        dataOffset = tensorDataStart;
        header = header.toBuilder().tensorCount(tensorInfos.size()).build();

        log.debug("Read {} tensors, data starts at offset {}", tensorInfos.size(), dataOffset);

        return tensorInfos;
    }

    /**
     * Read the raw data for a tensor
     */
    public byte[] readTensorData(GGMLTensorInfo tensorInfo) throws IOException {
        long absoluteOffset = dataOffset + tensorInfo.getDataOffset();
        long dataSize = tensorInfo.getDataSize();

        if (dataSize > Integer.MAX_VALUE) {
            throw new IOException("Tensor data too large: " + dataSize);
        }

        byte[] data = new byte[(int) dataSize];

        if (absoluteOffset + dataSize > buffer.capacity()) {
            channel.position(absoluteOffset);
            ByteBuffer tempBuffer = ByteBuffer.allocate((int) dataSize);
            channel.read(tempBuffer);
            tempBuffer.flip();
            tempBuffer.get(data);
        } else {
            buffer.position((int) absoluteOffset);
            buffer.get(data);
        }

        return data;
    }

    /**
     * Get the metadata from the header
     */
    public GGMLMetadata getMetadata() throws IOException, GGMLImportException {
        if (header == null) {
            readHeader();
        }
        if (tensorInfos == null) {
            readTensorInfos();
        }
        return GGMLMetadata.fromGGML(header, tensorInfos);
    }

    /**
     * Get the parsed header
     */
    public GGMLHeader getHeader() throws IOException, GGMLImportException {
        if (header == null) {
            readHeader();
        }
        return header;
    }

    /**
     * Get the list of tensor infos
     */
    public List<GGMLTensorInfo> getTensorInfos() throws IOException, GGMLImportException {
        if (tensorInfos == null) {
            readTensorInfos();
        }
        return tensorInfos;
    }

    // Private helper methods

    private GGMLHeader.LegacyFormat detectLegacyFormat(int magic) {
        // Check both byte orders
        if (magic == 0x67676D6C || magic == 0x6C6D6767) {
            return GGMLHeader.LegacyFormat.GGML;
        }
        if (magic == 0x67676D66 || magic == 0x666D6767) {
            return GGMLHeader.LegacyFormat.GGMF;
        }
        if (magic == 0x67676A74 || magic == 0x746A6767) {
            return GGMLHeader.LegacyFormat.GGJT;
        }
        return GGMLHeader.LegacyFormat.UNKNOWN;
    }

    private void skipVocabulary() throws IOException {
        if (header == null) return;

        // Skip vocabulary tokens (model-specific)
        // This is a simplified version - actual implementation depends on model type
        int vocabSize = header.getVocabSize();

        for (int i = 0; i < vocabSize; i++) {
            // Read token length and skip token bytes
            int tokenLen = buffer.getInt();
            if (tokenLen > 0 && tokenLen < 1000) {
                buffer.position(buffer.position() + tokenLen);
            }

            // Skip score for some models
            if (buffer.remaining() >= 4) {
                buffer.getFloat(); // token score
            }
        }
    }

    private static long alignOffset(long offset, int alignment) {
        return ((offset + alignment - 1) / alignment) * alignment;
    }

    @Override
    public void close() throws IOException {
        buffer = null;
        if (channel != null && channel.isOpen()) {
            channel.close();
        }
        if (raf != null) {
            raf.close();
        }
    }
}
