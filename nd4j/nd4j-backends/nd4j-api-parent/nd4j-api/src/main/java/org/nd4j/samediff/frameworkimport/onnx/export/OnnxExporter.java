/*
 *  ******************************************************************************
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

package org.nd4j.samediff.frameworkimport.onnx.export;

import org.nd4j.autodiff.samediff.SameDiff;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collections;
import java.util.List;

/**
 * ONNX exporter for SameDiff models.
 * Stub implementation - TODO: implement full functionality.
 */
public class OnnxExporter {

    /**
     * Export a SameDiff model to ONNX format.
     *
     * @param sameDiff The SameDiff model to export
     * @param file     The output file
     * @param config   Export configuration
     * @throws IOException If export fails
     */
    public static void export(SameDiff sameDiff, File file, OnnxExportConfig config) throws IOException {
        throw new UnsupportedOperationException("ONNX export not yet implemented");
    }

    /**
     * Export a SameDiff model to ONNX format.
     *
     * @param sameDiff     The SameDiff model to export
     * @param outputStream The output stream
     * @throws IOException If export fails
     */
    public static void export(SameDiff sameDiff, OutputStream outputStream) throws IOException {
        throw new UnsupportedOperationException("ONNX export not yet implemented");
    }

    /**
     * Export a SameDiff model to ONNX format with default configuration.
     *
     * @param sameDiff The SameDiff model to export
     * @param file     The output file
     * @throws IOException If export fails
     */
    public static void export(SameDiff sameDiff, File file) throws IOException {
        export(sameDiff, file, new OnnxExportConfig());
    }

    /**
     * Convert a SameDiff model to ONNX bytes.
     *
     * @param sameDiff The SameDiff model to convert
     * @return The ONNX model as bytes
     */
    public static byte[] toBytes(SameDiff sameDiff) {
        throw new UnsupportedOperationException("ONNX export not yet implemented");
    }

    /**
     * Check if a SameDiff model can be exported to ONNX.
     *
     * @param sameDiff The SameDiff model to check
     * @return true if the model can be exported
     */
    public static boolean canExport(SameDiff sameDiff) {
        return false; // Stub - export not implemented
    }

    /**
     * Validate a SameDiff model for ONNX export.
     *
     * @param sameDiff The SameDiff model to validate
     * @return List of validation errors, empty if valid
     */
    public static List<String> validateForExport(SameDiff sameDiff) {
        return Collections.singletonList("ONNX export not yet implemented");
    }
}
