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

package org.eclipse.deeplearning4j.ggml.format;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.ggml.GGMLImportException;
import org.nd4j.ggml.format.GGMLFormat;
import org.nd4j.ggml.format.GGMLFormatDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("GGMLFormatDetector Test")
class GGMLFormatDetectorTest {

    @TempDir
    Path tempDir;

    @Test
    @DisplayName("Test detect GGUF format")
    void testDetectGGUF() throws IOException, GGMLImportException {
        File ggufFile = createGGUFFile();
        GGMLFormat format = GGMLFormatDetector.detect(ggufFile);
        assertEquals(GGMLFormat.GGUF, format);
    }

    @Test
    @DisplayName("Test detect legacy GGML format")
    void testDetectGGML() throws IOException, GGMLImportException {
        File ggmlFile = createLegacyGGMLFile(0x67676D6C); // 'ggml' in little-endian
        GGMLFormat format = GGMLFormatDetector.detect(ggmlFile);
        assertEquals(GGMLFormat.GGML, format);
    }

    @Test
    @DisplayName("Test detect legacy GGMF format")
    void testDetectGGMF() throws IOException, GGMLImportException {
        File ggmfFile = createLegacyGGMLFile(0x67676D66); // 'ggmf' in little-endian
        GGMLFormat format = GGMLFormatDetector.detect(ggmfFile);
        assertEquals(GGMLFormat.GGML, format);
    }

    @Test
    @DisplayName("Test detect legacy GGJT format")
    void testDetectGGJT() throws IOException, GGMLImportException {
        File ggjtFile = createLegacyGGMLFile(0x67676A74); // 'ggjt' in little-endian
        GGMLFormat format = GGMLFormatDetector.detect(ggjtFile);
        assertEquals(GGMLFormat.GGML, format);
    }

    @Test
    @DisplayName("Test detect throws for invalid format")
    void testDetectInvalidFormat() throws IOException {
        File invalidFile = tempDir.resolve("invalid.bin").toFile();
        try (FileOutputStream fos = new FileOutputStream(invalidFile)) {
            fos.write(new byte[]{0x00, 0x00, 0x00, 0x00});
        }
        assertThrows(GGMLImportException.class, () -> GGMLFormatDetector.detect(invalidFile));
    }

    @Test
    @DisplayName("Test detect throws for non-existent file")
    void testDetectNonExistentFile() {
        File nonExistent = new File("/non/existent/file.gguf");
        assertThrows(Exception.class, () -> GGMLFormatDetector.detect(nonExistent));
    }

    @Test
    @DisplayName("Test detect throws for file too small")
    void testDetectFileTooSmall() throws IOException {
        File smallFile = tempDir.resolve("small.bin").toFile();
        try (FileOutputStream fos = new FileOutputStream(smallFile)) {
            fos.write(new byte[]{0x47}); // Only 1 byte
        }
        assertThrows(GGMLImportException.class, () -> GGMLFormatDetector.detect(smallFile));
    }

    private File createGGUFFile() throws IOException {
        File file = tempDir.resolve("test.gguf").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            // GGUF magic: 'GGUF' = 0x46554747 in little-endian
            buffer.putInt(0x46554747);
            // Version
            buffer.putInt(3);
            // Tensor count
            buffer.putLong(0);
            // Metadata KV count
            buffer.putLong(0);
            fos.write(buffer.array());
        }
        return file;
    }

    private File createLegacyGGMLFile(int magic) throws IOException {
        File file = tempDir.resolve("test.ggml").toFile();
        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(8);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            buffer.putInt(magic);
            buffer.putInt(1); // version
            fos.write(buffer.array());
        }
        return file;
    }
}
