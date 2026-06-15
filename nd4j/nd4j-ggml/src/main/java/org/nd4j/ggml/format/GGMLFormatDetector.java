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

import org.nd4j.ggml.GGMLImportException;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Detects the format of GGML/GGUF files by examining magic bytes.
 */
public class GGMLFormatDetector {

    // GGUF magic: 'GGUF' in little-endian (0x46554747)
    public static final int GGUF_MAGIC = 0x46554747;

    // Legacy GGML magic patterns
    public static final int GGML_MAGIC_GGML = 0x67676D6C; // 'ggml'
    public static final int GGML_MAGIC_GGMF = 0x67676D66; // 'ggmf'
    public static final int GGML_MAGIC_GGJT = 0x67676A74; // 'ggjt'

    private GGMLFormatDetector() {
        // Utility class
    }

    /**
     * Detect the format of a GGML/GGUF file.
     *
     * @param file the file to examine
     * @return the detected format
     * @throws IOException if the file cannot be read
     * @throws GGMLImportException if the format cannot be detected
     */
    public static GGMLFormat detect(File file) throws IOException, GGMLImportException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            if (raf.length() < 4) {
                throw new GGMLImportException("File too small to be a valid GGML/GGUF file: " + file);
            }

            byte[] magicBytes = new byte[4];
            raf.readFully(magicBytes);

            return detectFromMagic(magicBytes, file.getName());
        }
    }

    /**
     * Detect the format from an input stream.
     * Note: This reads 4 bytes from the stream and does not reset it.
     *
     * @param is the input stream
     * @return the detected format
     * @throws IOException if the stream cannot be read
     * @throws GGMLImportException if the format cannot be detected
     */
    public static GGMLFormat detect(InputStream is) throws IOException, GGMLImportException {
        byte[] magicBytes = new byte[4];
        int bytesRead = is.read(magicBytes);
        if (bytesRead < 4) {
            throw new GGMLImportException("Could not read magic bytes from stream");
        }

        return detectFromMagic(magicBytes, "stream");
    }

    /**
     * Detect the format from magic bytes.
     *
     * @param magicBytes the first 4 bytes of the file
     * @param sourceName name of the source for error messages
     * @return the detected format
     * @throws GGMLImportException if the format cannot be detected
     */
    public static GGMLFormat detectFromMagic(byte[] magicBytes, String sourceName) throws GGMLImportException {
        if (magicBytes.length < 4) {
            throw new GGMLImportException("Magic bytes array too short");
        }

        // Read as little-endian int
        ByteBuffer buffer = ByteBuffer.wrap(magicBytes).order(ByteOrder.LITTLE_ENDIAN);
        int magic = buffer.getInt();

        if (magic == GGUF_MAGIC) {
            return GGMLFormat.GGUF;
        }

        // Check for legacy GGML formats (big-endian magic)
        buffer.rewind();
        buffer.order(ByteOrder.BIG_ENDIAN);
        int magicBE = buffer.getInt();

        if (magicBE == GGML_MAGIC_GGML || magicBE == GGML_MAGIC_GGMF || magicBE == GGML_MAGIC_GGJT) {
            return GGMLFormat.GGML;
        }

        // Also check little-endian for legacy formats
        if (magic == GGML_MAGIC_GGML || magic == GGML_MAGIC_GGMF || magic == GGML_MAGIC_GGJT) {
            return GGMLFormat.GGML;
        }

        throw new GGMLImportException(
            String.format("Unknown file format. Magic bytes: 0x%08X (source: %s)", magic, sourceName));
    }

    /**
     * Check if a file appears to be a GGUF file based on extension.
     *
     * @param filename the filename to check
     * @return true if the filename suggests GGUF format
     */
    public static boolean hasGGUFExtension(String filename) {
        String lower = filename.toLowerCase();
        return lower.endsWith(".gguf");
    }

    /**
     * Check if a file appears to be a legacy GGML file based on extension.
     *
     * @param filename the filename to check
     * @return true if the filename suggests legacy GGML format
     */
    public static boolean hasGGMLExtension(String filename) {
        String lower = filename.toLowerCase();
        return lower.endsWith(".ggml") || lower.endsWith(".bin");
    }

    /**
     * Get the GGUF magic number as a string.
     */
    public static String getGGUFMagicString() {
        return "GGUF";
    }

    /**
     * Get the GGUF version from a file.
     *
     * @param file the GGUF file
     * @return the GGUF version number
     * @throws IOException if the file cannot be read
     * @throws GGMLImportException if the file is not a valid GGUF file
     */
    public static int getGGUFVersion(File file) throws IOException, GGMLImportException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            if (raf.length() < 8) {
                throw new GGMLImportException("File too small to contain GGUF version");
            }

            byte[] headerBytes = new byte[8];
            raf.readFully(headerBytes);

            ByteBuffer buffer = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN);
            int magic = buffer.getInt();

            if (magic != GGUF_MAGIC) {
                throw new GGMLImportException("Not a GGUF file");
            }

            return buffer.getInt();
        }
    }
}
