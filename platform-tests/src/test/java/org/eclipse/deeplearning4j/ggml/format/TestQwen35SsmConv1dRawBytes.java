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

import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.DownloadResult;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.LLMModel;
import org.eclipse.deeplearning4j.llm.data.LLMModelDownloader.QuantType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.format.GGMLTensorInfo;
import org.nd4j.ggml.format.GGUFReader;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Minimal test to verify raw GGUF bytes for ssm_conv1d.weight tensors
 * in the Qwen3.5-0.8B Q4_K_M model file.
 *
 * This reads directly from the GGUF file without any dequantization
 * to check if the raw data contains non-zero bytes.
 */
public class TestQwen35SsmConv1dRawBytes {

    @Test
    @DisplayName("Verify ssm_conv1d.weight raw GGUF bytes are non-zero")
    public void testSsmConv1dWeightRawBytes() throws Exception {
        // 1. Download model
        System.out.println("=== Downloading Qwen3.5-0.8B Q4_K_M ===");
        DownloadResult result = LLMModelDownloader.download(LLMModel.QWEN35_0_8B, QuantType.Q4_K_M);
        File ggufFile = result.getModelFile();
        System.out.println("Model file: " + ggufFile.getAbsolutePath());
        System.out.println("File size: " + ggufFile.length() + " bytes");
        assertTrue(ggufFile.exists(), "Model file must exist");

        // 2. Read tensor infos
        System.out.println("\n=== Reading GGUF tensor infos ===");
        boolean foundAny = false;
        boolean anyAllZero = false;

        try (GGUFReader reader = new GGUFReader(ggufFile)) {
            List<GGMLTensorInfo> tensors = reader.getTensorInfos();
            System.out.println("Total tensors in file: " + tensors.size());

            for (GGMLTensorInfo ti : tensors) {
                if (!ti.getName().contains("ssm_conv1d")) {
                    continue;
                }

                foundAny = true;
                System.out.println("\n--- Tensor: " + ti.getName() + " ---");
                System.out.println("  DataType: " + ti.getDataType());
                System.out.println("  Shape: " + Arrays.toString(ti.getShape()));
                System.out.println("  NumElements: " + ti.getNumElements());
                System.out.println("  DataSize: " + ti.getDataSize() + " bytes");
                System.out.println("  DataOffset: " + ti.getDataOffset());

                // 3. Read raw bytes
                byte[] rawData = reader.readTensorData(ti);
                System.out.println("  Raw data length: " + rawData.length + " bytes");

                // 4. Print first 40 bytes as hex
                int printLen = Math.min(40, rawData.length);
                StringBuilder hex = new StringBuilder();
                for (int i = 0; i < printLen; i++) {
                    hex.append(String.format("%02X ", rawData[i] & 0xFF));
                }
                System.out.println("  First " + printLen + " bytes (hex): " + hex.toString().trim());

                // Also print last 40 bytes
                int lastStart = Math.max(0, rawData.length - 40);
                StringBuilder hexLast = new StringBuilder();
                for (int i = lastStart; i < rawData.length; i++) {
                    hexLast.append(String.format("%02X ", rawData[i] & 0xFF));
                }
                System.out.println("  Last " + (rawData.length - lastStart) + " bytes (hex): " + hexLast.toString().trim());

                // 5. Check if all bytes are zero
                boolean allZero = true;
                int nonZeroCount = 0;
                for (byte b : rawData) {
                    if (b != 0) {
                        allZero = false;
                        nonZeroCount++;
                    }
                }
                System.out.println("  All zero: " + allZero);
                System.out.println("  Non-zero byte count: " + nonZeroCount + " / " + rawData.length
                        + " (" + String.format("%.1f%%", 100.0 * nonZeroCount / rawData.length) + ")");

                if (allZero) {
                    anyAllZero = true;
                    System.out.println("  *** WARNING: ALL BYTES ARE ZERO ***");
                }
            }
        }

        assertTrue(foundAny, "Should find at least one ssm_conv1d tensor in model");
        assertFalse(anyAllZero, "No ssm_conv1d.weight tensor should be all-zero in raw GGUF data");
    }
}
