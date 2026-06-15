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

package org.nd4j.torchscript.format;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.torchscript.TorchScriptImportException;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Reader for Safetensors files.
 *
 * <p>Safetensors format:
 * <ul>
 *   <li>8 bytes: header size (little-endian u64)</li>
 *   <li>N bytes: JSON header with tensor metadata</li>
 *   <li>Remaining: Raw tensor data</li>
 * </ul>
 */
@Slf4j
public class SafetensorsReader implements Closeable {

    private final File file;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private MappedByteBuffer dataBuffer;

    private SafetensorsHeader header;
    private long headerSize;
    private long dataOffset;

    /**
     * Create a new Safetensors reader.
     *
     * @param file the Safetensors file
     * @throws IOException if file cannot be opened
     */
    public SafetensorsReader(File file) throws IOException {
        this.file = file;
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();
    }

    /**
     * Read and parse the header.
     *
     * @return the parsed header
     * @throws TorchScriptImportException if parsing fails
     */
    public SafetensorsHeader readHeader() throws TorchScriptImportException {
        if (header != null) {
            return header;
        }

        try {
            // Read header size (8 bytes, little-endian u64)
            ByteBuffer sizeBuffer = ByteBuffer.allocate(8);
            sizeBuffer.order(ByteOrder.LITTLE_ENDIAN);
            channel.position(0);
            channel.read(sizeBuffer);
            sizeBuffer.flip();
            headerSize = sizeBuffer.getLong();

            log.debug("Header size: {} bytes", headerSize);

            // Validate header size
            if (headerSize <= 0 || headerSize > file.length() - 8) {
                throw new TorchScriptImportException(
                        "Invalid header size: " + headerSize + " (file size: " + file.length() + ")");
            }

            // Read header JSON
            ByteBuffer headerBuffer = ByteBuffer.allocate((int) headerSize);
            channel.position(8);
            channel.read(headerBuffer);
            headerBuffer.flip();
            String headerJson = StandardCharsets.UTF_8.decode(headerBuffer).toString();

            log.debug("Header JSON length: {} chars", headerJson.length());

            // Parse header
            header = SafetensorsHeader.parse(headerJson);

            // Calculate data section offset
            dataOffset = 8 + headerSize;

            log.info("Loaded Safetensors header with {} tensors", header.getTensorCount());

            return header;

        } catch (IOException e) {
            throw new TorchScriptImportException("Failed to read Safetensors header", e);
        }
    }

    /**
     * Get tensor info list from the header.
     *
     * @return list of tensor information
     * @throws TorchScriptImportException if header not read
     */
    public List<TorchScriptTensorInfo> readTensorInfos() throws TorchScriptImportException {
        if (header == null) {
            readHeader();
        }
        return header.toTensorInfoList(dataOffset);
    }

    /**
     * Read tensor data for a specific tensor.
     *
     * @param tensorInfo the tensor to read
     * @return raw tensor data bytes
     * @throws IOException if reading fails
     */
    public byte[] readTensorData(TorchScriptTensorInfo tensorInfo) throws IOException {
        long offset = tensorInfo.getDataOffset();
        long size = tensorInfo.getDataSize();

        log.debug("Reading tensor {} at offset {} with size {}", tensorInfo.getName(), offset, size);

        // Validate
        if (offset < dataOffset || offset + size > file.length()) {
            throw new IOException("Invalid tensor data range: offset=" + offset +
                    ", size=" + size + ", file length=" + file.length());
        }

        // Read data
        ByteBuffer buffer = ByteBuffer.allocate((int) size);
        channel.position(offset);
        int bytesRead = channel.read(buffer);

        if (bytesRead != size) {
            throw new IOException("Expected to read " + size + " bytes but got " + bytesRead);
        }

        buffer.flip();
        byte[] data = new byte[(int) size];
        buffer.get(data);
        return data;
    }

    /**
     * Get a memory-mapped view of tensor data.
     *
     * @param tensorInfo the tensor to map
     * @return mapped byte buffer
     * @throws IOException if mapping fails
     */
    public MappedByteBuffer mapTensorData(TorchScriptTensorInfo tensorInfo) throws IOException {
        long offset = tensorInfo.getDataOffset();
        long size = tensorInfo.getDataSize();

        return channel.map(FileChannel.MapMode.READ_ONLY, offset, size);
    }

    /**
     * Get metadata for the model.
     *
     * @return model metadata
     * @throws TorchScriptImportException if header not read
     */
    public TorchScriptMetadata getMetadata() throws TorchScriptImportException {
        if (header == null) {
            readHeader();
        }

        List<TorchScriptTensorInfo> tensors = readTensorInfos();

        return TorchScriptMetadata.builder()
                .format(TorchScriptFormat.SAFETENSORS)
                .tensors(tensors)
                .rawMetadata(header.getMetadata())
                .build();
    }

    /**
     * Get the data section offset.
     *
     * @return offset in bytes where tensor data begins
     */
    public long getDataOffset() {
        return dataOffset;
    }

    /**
     * Get the file being read.
     *
     * @return the file
     */
    public File getFile() {
        return file;
    }

    @Override
    public void close() throws IOException {
        if (dataBuffer != null) {
            // MappedByteBuffer doesn't have explicit close, relies on GC
            dataBuffer = null;
        }
        if (channel != null) {
            channel.close();
        }
        if (raf != null) {
            raf.close();
        }
    }
}
