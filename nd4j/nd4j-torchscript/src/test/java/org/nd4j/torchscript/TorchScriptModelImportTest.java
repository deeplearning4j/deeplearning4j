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

package org.nd4j.torchscript;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.torchscript.format.TorchScriptFormat;
import org.nd4j.torchscript.format.TorchScriptMetadata;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("TorchScriptModelImport Test")
class TorchScriptModelImportTest {

    @TempDir
    File tempDir;

    @Test
    @DisplayName("Test isSupportedFile for safetensors")
    void testIsSupportedFileSafetensors() throws IOException {
        File safetensorsFile = createSafetensorsFile();
        assertTrue(TorchScriptModelImport.isSupportedFile(safetensorsFile));
        assertTrue(TorchScriptModelImport.isSupportedFile(safetensorsFile.getAbsolutePath()));
    }

    @Test
    @DisplayName("Test isSupportedFile returns false for unknown format")
    void testIsSupportedFileUnknown() throws IOException {
        File unknownFile = new File(tempDir, "unknown.dat");
        try (FileOutputStream fos = new FileOutputStream(unknownFile)) {
            fos.write(new byte[]{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09});
        }
        assertFalse(TorchScriptModelImport.isSupportedFile(unknownFile));
    }

    @Test
    @DisplayName("Test isSupportedFile returns false for non-existent file")
    void testIsSupportedFileNonExistent() {
        assertFalse(TorchScriptModelImport.isSupportedFile("/non/existent/file.safetensors"));
    }

    @Test
    @DisplayName("Test detectFormat for safetensors")
    void testDetectFormatSafetensors() throws IOException, TorchScriptImportException {
        File safetensorsFile = createSafetensorsFile();
        TorchScriptFormat format = TorchScriptModelImport.detectFormat(safetensorsFile);
        assertEquals(TorchScriptFormat.SAFETENSORS, format);
    }

    @Test
    @DisplayName("Test detectFormat by path")
    void testDetectFormatByPath() throws IOException, TorchScriptImportException {
        File safetensorsFile = createSafetensorsFile();
        TorchScriptFormat format = TorchScriptModelImport.detectFormat(safetensorsFile.getAbsolutePath());
        assertEquals(TorchScriptFormat.SAFETENSORS, format);
    }

    @Test
    @DisplayName("Test detectFormat throws for non-existent file")
    void testDetectFormatNonExistent() {
        assertThrows(TorchScriptImportException.class,
                () -> TorchScriptModelImport.detectFormat("/non/existent/file.pt"));
    }

    @Test
    @DisplayName("Test inspectModel for safetensors")
    void testInspectModelSafetensors() throws IOException, TorchScriptImportException {
        File safetensorsFile = createSafetensorsFileWithMultipleTensors();
        TorchScriptMetadata metadata = TorchScriptModelImport.inspectModel(safetensorsFile);

        assertNotNull(metadata);
        assertEquals(TorchScriptFormat.SAFETENSORS, metadata.getFormat());
        assertEquals(2, metadata.getTensors().size());
        assertTrue(metadata.getTotalParameters() > 0);
    }

    @Test
    @DisplayName("Test inspectModel by path")
    void testInspectModelByPath() throws IOException, TorchScriptImportException {
        File safetensorsFile = createSafetensorsFile();
        TorchScriptMetadata metadata = TorchScriptModelImport.inspectModel(safetensorsFile.getAbsolutePath());

        assertNotNull(metadata);
    }

    @Test
    @DisplayName("Test inspectModel throws for non-existent file")
    void testInspectModelNonExistent() {
        assertThrows(TorchScriptImportException.class,
                () -> TorchScriptModelImport.inspectModel("/non/existent/file.safetensors"));
    }

    @Test
    @DisplayName("Test importModel throws for non-existent file")
    void testImportModelNonExistent() {
        assertThrows(TorchScriptImportException.class,
                () -> TorchScriptModelImport.importModel("/non/existent/file.pt"));
    }

    @Test
    @DisplayName("Test convertToSDZ throws for non-existent file")
    void testConvertToSDZNonExistent() {
        File outputFile = new File(tempDir, "output.sdz");
        assertThrows(TorchScriptImportException.class,
                () -> TorchScriptModelImport.convertToSDZ("/non/existent/file.safetensors", outputFile.getAbsolutePath()));
    }

    // Helper methods

    private File createSafetensorsFile() throws IOException {
        File file = new File(tempDir, "test.safetensors");

        String header = "{\"test_tensor\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(8 + headerBytes.length + 24);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            buffer.putLong(headerBytes.length);
            buffer.put(headerBytes);

            // Write 6 floats
            for (int i = 0; i < 6; i++) {
                buffer.putFloat(i * 1.0f);
            }

            fos.write(buffer.array());
        }

        return file;
    }

    private File createSafetensorsFileWithMultipleTensors() throws IOException {
        File file = new File(tempDir, "multi_tensor.safetensors");

        String header = "{" +
                "\"layer1.weight\":{\"dtype\":\"F32\",\"shape\":[4,4],\"data_offsets\":[0,64]}," +
                "\"layer1.bias\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[64,80]}" +
                "}";
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        try (FileOutputStream fos = new FileOutputStream(file)) {
            ByteBuffer buffer = ByteBuffer.allocate(8 + headerBytes.length + 80);
            buffer.order(ByteOrder.LITTLE_ENDIAN);

            buffer.putLong(headerBytes.length);
            buffer.put(headerBytes);

            // Write 20 floats (16 + 4)
            for (int i = 0; i < 20; i++) {
                buffer.putFloat(i * 0.1f);
            }

            fos.write(buffer.array());
        }

        return file;
    }
}
