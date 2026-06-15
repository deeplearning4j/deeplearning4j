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

// TODO: Move to platform-tests/ per project convention — all tests must run from platform-tests

package org.nd4j.torchscript.format;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("TorchScriptFormatDetector Test")
class TorchScriptFormatDetectorTest {

    @TempDir
    File tempDir;

    @Test
    @DisplayName("Test detect Safetensors format")
    void testDetectSafetensors() throws IOException {
        File safetensorsFile = createSafetensorsFile();
        TorchScriptFormat format = TorchScriptFormatDetector.detect(safetensorsFile);
        assertEquals(TorchScriptFormat.SAFETENSORS, format);
    }

    @Test
    @DisplayName("Test detect TorchScript PT format")
    void testDetectTorchScriptPT() throws IOException {
        File ptFile = createTorchScriptZipFile();
        TorchScriptFormat format = TorchScriptFormatDetector.detect(ptFile);
        assertEquals(TorchScriptFormat.TORCHSCRIPT_PT, format);
    }

    @Test
    @DisplayName("Test detect PyTorch BIN format")
    void testDetectPytorchBin() throws IOException {
        File binFile = createPickleFile();
        TorchScriptFormat format = TorchScriptFormatDetector.detect(binFile);
        assertEquals(TorchScriptFormat.PYTORCH_BIN, format);
    }

    @Test
    @DisplayName("Test detect unknown format")
    void testDetectUnknownFormat() throws IOException {
        File unknownFile = new File(tempDir, "unknown.dat");
        try (FileOutputStream fos = new FileOutputStream(unknownFile)) {
            fos.write(new byte[]{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09});
        }
        TorchScriptFormat format = TorchScriptFormatDetector.detect(unknownFile);
        assertEquals(TorchScriptFormat.UNKNOWN, format);
    }

    @Test
    @DisplayName("Test detect throws for non-existent file")
    void testDetectNonExistentFile() {
        File nonExistent = new File("/non/existent/file.pt");
        assertThrows(IOException.class, () -> TorchScriptFormatDetector.detect(nonExistent));
    }

    @Test
    @DisplayName("Test detect file too small")
    void testDetectFileTooSmall() throws IOException {
        File smallFile = new File(tempDir, "small.bin");
        try (FileOutputStream fos = new FileOutputStream(smallFile)) {
            fos.write(new byte[]{0x01, 0x02});
        }
        TorchScriptFormat format = TorchScriptFormatDetector.detect(smallFile);
        assertEquals(TorchScriptFormat.UNKNOWN, format);
    }

    @Test
    @DisplayName("Test isSafetensors")
    void testIsSafetensors() throws IOException {
        File safetensorsFile = createSafetensorsFile();
        assertTrue(TorchScriptFormatDetector.isSafetensors(safetensorsFile));

        File notSafetensors = new File(tempDir, "not_safetensors.bin");
        try (FileOutputStream fos = new FileOutputStream(notSafetensors)) {
            fos.write(new byte[20]);
        }
        assertFalse(TorchScriptFormatDetector.isSafetensors(notSafetensors));
    }

    @Test
    @DisplayName("Test isSafetensors with file too small")
    void testIsSafetensorsFileTooSmall() throws IOException {
        File smallFile = new File(tempDir, "small_safetensors.bin");
        try (FileOutputStream fos = new FileOutputStream(smallFile)) {
            fos.write(new byte[10]);
        }
        assertFalse(TorchScriptFormatDetector.isSafetensors(smallFile));
    }

    @Test
    @DisplayName("Test isZipFile")
    void testIsZipFile() throws IOException {
        File zipFile = createTorchScriptZipFile();
        assertTrue(TorchScriptFormatDetector.isZipFile(zipFile));

        File notZip = new File(tempDir, "not_zip.bin");
        try (FileOutputStream fos = new FileOutputStream(notZip)) {
            fos.write(new byte[]{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07});
        }
        assertFalse(TorchScriptFormatDetector.isZipFile(notZip));
    }

    @Test
    @DisplayName("Test isTorchScriptZip")
    void testIsTorchScriptZip() throws IOException {
        File ptFile = createTorchScriptZipFile();
        assertTrue(TorchScriptFormatDetector.isTorchScriptZip(ptFile));

        // Create a regular ZIP file without TorchScript structure
        File regularZip = new File(tempDir, "regular.zip");
        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(regularZip))) {
            ZipEntry entry = new ZipEntry("random_file.txt");
            zos.putNextEntry(entry);
            zos.write("Hello World".getBytes(StandardCharsets.UTF_8));
            zos.closeEntry();
        }
        assertFalse(TorchScriptFormatDetector.isTorchScriptZip(regularZip));
    }

    @Test
    @DisplayName("Test isPickleFile")
    void testIsPickleFile() throws IOException {
        File pickleFile = createPickleFile();
        assertTrue(TorchScriptFormatDetector.isPickleFile(pickleFile));

        File notPickle = new File(tempDir, "not_pickle.bin");
        try (FileOutputStream fos = new FileOutputStream(notPickle)) {
            fos.write(new byte[]{0x00, 0x01, 0x02, 0x03});
        }
        assertFalse(TorchScriptFormatDetector.isPickleFile(notPickle));
    }

    @Test
    @DisplayName("Test getExtension")
    void testGetExtension() {
        assertEquals(".pt", TorchScriptFormatDetector.getExtension(TorchScriptFormat.TORCHSCRIPT_PT));
        assertEquals(".safetensors", TorchScriptFormatDetector.getExtension(TorchScriptFormat.SAFETENSORS));
        assertEquals(".bin", TorchScriptFormatDetector.getExtension(TorchScriptFormat.PYTORCH_BIN));
        assertEquals("", TorchScriptFormatDetector.getExtension(TorchScriptFormat.UNKNOWN));
    }

    // Helper methods to create test files

    private File createSafetensorsFile() throws IOException {
        File file = new File(tempDir, "test.safetensors");
        try (FileOutputStream fos = new FileOutputStream(file)) {
            // Create a minimal valid safetensors file
            String header = "{\"test_tensor\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}";
            byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

            ByteBuffer buffer = ByteBuffer.allocate(8 + headerBytes.length + 24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            // Header size (8 bytes, little-endian)
            buffer.putLong(headerBytes.length);

            // Header JSON
            buffer.put(headerBytes);

            // Dummy tensor data (6 floats = 24 bytes)
            for (int i = 0; i < 6; i++) {
                buffer.putFloat(i * 1.0f);
            }

            fos.write(buffer.array());
        }
        return file;
    }

    private File createTorchScriptZipFile() throws IOException {
        File file = new File(tempDir, "test.pt");
        try (ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(file))) {
            // Add data.pkl entry (empty for testing)
            ZipEntry dataPkl = new ZipEntry("data.pkl");
            zos.putNextEntry(dataPkl);
            // Write minimal pickle data (protocol + stop opcode)
            zos.write(new byte[]{(byte) 0x80, 0x02, (byte) 0x2E});
            zos.closeEntry();

            // Add constants.pkl entry
            ZipEntry constantsPkl = new ZipEntry("constants.pkl");
            zos.putNextEntry(constantsPkl);
            zos.write(new byte[]{(byte) 0x80, 0x02, (byte) 0x2E});
            zos.closeEntry();
        }
        return file;
    }

    private File createPickleFile() throws IOException {
        File file = new File(tempDir, "test.bin");
        try (FileOutputStream fos = new FileOutputStream(file)) {
            // Pickle protocol 4 header + stop opcode
            fos.write(new byte[]{
                    (byte) 0x80, 0x04,  // Protocol marker + version 4
                    (byte) 0x95,        // FRAME opcode
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Frame size
                    (byte) 0x2E        // STOP opcode
            });
        }
        return file;
    }
}
