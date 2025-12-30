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

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.zip.ZipFile;

/**
 * Detects the format of TorchScript and related model files.
 */
@Slf4j
public class TorchScriptFormatDetector {

    // ZIP magic bytes (TorchScript .pt files are ZIP archives)
    private static final byte[] ZIP_MAGIC = {0x50, 0x4B, 0x03, 0x04};

    // Pickle protocol marker (legacy .bin files)
    private static final byte PICKLE_PROTO = (byte) 0x80;

    private TorchScriptFormatDetector() {
        // Utility class
    }

    /**
     * Detect the format of a model file.
     *
     * @param file the file to check
     * @return the detected format
     * @throws IOException if file cannot be read
     */
    public static TorchScriptFormat detect(File file) throws IOException {
        if (!file.exists()) {
            throw new IOException("File not found: " + file);
        }

        if (file.length() < 8) {
            return TorchScriptFormat.UNKNOWN;
        }

        // Check for Safetensors format first (has distinct header structure)
        if (isSafetensors(file)) {
            return TorchScriptFormat.SAFETENSORS;
        }

        // Check for ZIP format (TorchScript .pt files)
        if (isZipFile(file)) {
            if (isTorchScriptZip(file)) {
                return TorchScriptFormat.TORCHSCRIPT_PT;
            }
        }

        // Check for pickle format (legacy .bin files)
        if (isPickleFile(file)) {
            return TorchScriptFormat.PYTORCH_BIN;
        }

        return TorchScriptFormat.UNKNOWN;
    }

    /**
     * Check if a file is a Safetensors file.
     * Safetensors files have a 64-bit header size followed by valid JSON.
     *
     * @param file the file to check
     * @return true if this appears to be a Safetensors file
     * @throws IOException if file cannot be read
     */
    public static boolean isSafetensors(File file) throws IOException {
        if (file.length() < 16) {
            return false;
        }

        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel channel = raf.getChannel()) {

            ByteBuffer buffer = ByteBuffer.allocate(16);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(buffer);
            buffer.flip();

            // First 8 bytes are header size (little-endian u64)
            long headerSize = buffer.getLong();

            // Sanity check: header size should be reasonable
            // and file should be large enough to contain header + data
            if (headerSize <= 0 || headerSize > 100_000_000) { // 100MB max header
                return false;
            }

            if (file.length() < 8 + headerSize) {
                return false;
            }

            // Read first byte of header - should be '{' for JSON
            byte firstByte = buffer.get();
            return firstByte == '{';
        }
    }

    /**
     * Check if a file is a ZIP file.
     *
     * @param file the file to check
     * @return true if this is a ZIP file
     * @throws IOException if file cannot be read
     */
    public static boolean isZipFile(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            byte[] magic = new byte[4];
            raf.readFully(magic);
            return magic[0] == ZIP_MAGIC[0] && magic[1] == ZIP_MAGIC[1] &&
                   magic[2] == ZIP_MAGIC[2] && magic[3] == ZIP_MAGIC[3];
        }
    }

    /**
     * Check if a ZIP file is a TorchScript model.
     * TorchScript ZIP files contain data.pkl and optionally code/ directory.
     *
     * @param file the ZIP file to check
     * @return true if this is a TorchScript ZIP
     * @throws IOException if file cannot be read
     */
    public static boolean isTorchScriptZip(File file) throws IOException {
        try (ZipFile zipFile = new ZipFile(file)) {
            // TorchScript files contain data.pkl or model.pkl
            boolean hasDataPkl = zipFile.getEntry("data.pkl") != null;
            boolean hasModelPkl = zipFile.getEntry("model.pkl") != null;
            boolean hasArchive = zipFile.getEntry("archive/data.pkl") != null;

            // Also check for typical TorchScript structure
            boolean hasConstants = zipFile.getEntry("constants.pkl") != null;
            boolean hasCode = zipFile.stream().anyMatch(e -> e.getName().startsWith("code/"));
            boolean hasData = zipFile.stream().anyMatch(e -> e.getName().startsWith("data/"));

            return hasDataPkl || hasModelPkl || hasArchive ||
                   (hasConstants && (hasCode || hasData));
        } catch (Exception e) {
            log.debug("Failed to check ZIP structure: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Check if a file is a Python pickle file.
     *
     * @param file the file to check
     * @return true if this appears to be a pickle file
     * @throws IOException if file cannot be read
     */
    public static boolean isPickleFile(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            byte firstByte = raf.readByte();
            // Pickle protocol 2+ starts with 0x80 followed by protocol version
            if (firstByte == PICKLE_PROTO) {
                byte version = raf.readByte();
                return version >= 2 && version <= 5;
            }
            return false;
        }
    }

    /**
     * Get the file extension for a format.
     *
     * @param format the format
     * @return typical file extension
     */
    public static String getExtension(TorchScriptFormat format) {
        switch (format) {
            case TORCHSCRIPT_PT:
                return ".pt";
            case SAFETENSORS:
                return ".safetensors";
            case PYTORCH_BIN:
                return ".bin";
            default:
                return "";
        }
    }
}
