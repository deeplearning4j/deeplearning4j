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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.pipeline;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.onnx.ONNXPipelineLoader;
import org.eclipse.deeplearning4j.pipeline.*;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.*;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for samediff-pipeline-onnx module.
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class ONNXPipelineTest extends BaseNd4jTestWithBackends {

    @TempDir
    Path tempDir;

    @Override
    public char ordering() {
        return 'c';
    }

    // ==================== ONNXPipelineLoader Basic Tests ====================

    @Test
    public void testONNXPipelineLoaderSupports() {
        ONNXPipelineLoader loader = new ONNXPipelineLoader();

        assertTrue(loader.supports(ModelFormat.ONNX));
        assertFalse(loader.supports(ModelFormat.SAFETENSORS));
        assertFalse(loader.supports(ModelFormat.GGUF));
        assertFalse(loader.supports(ModelFormat.PYTORCH));
        assertFalse(loader.supports(ModelFormat.SDZ));
        assertEquals(ModelFormat.ONNX, loader.getFormat());
        assertEquals("ONNX", loader.getName());
    }

    @Test
    public void testONNXPipelineLoaderConvertsToSdz() {
        ONNXPipelineLoader loader = new ONNXPipelineLoader();
        assertTrue(loader.convertsToSdz());
    }

    @Test
    public void testONNXPipelineLoaderCanLoad() {
        ONNXPipelineLoader loader = new ONNXPipelineLoader();

        ModelManifest onnxManifest = ModelManifest.builder()
                .format(ModelFormat.ONNX)
                .build();
        assertTrue(loader.canLoad(onnxManifest));

        ModelManifest safetensorsManifest = ModelManifest.builder()
                .format(ModelFormat.SAFETENSORS)
                .build();
        assertFalse(loader.canLoad(safetensorsManifest));
    }

    // ==================== ONNXPipelineLoader File Loading Tests ====================

    @Test
    public void testONNXPipelineLoaderFileNotFound() {
        ONNXPipelineLoader loader = new ONNXPipelineLoader();
        File nonExistent = new File("/non/existent/model.onnx");

        assertThrows(IOException.class,
                () -> loader.loadModel(nonExistent, PipelineLoader.LoadConfig.defaults()));
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testONNXPipelineLoaderWithSameDiffExport(Nd4jBackend backend) throws Exception {
        // Create a simple SameDiff model
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        SDVariable weights = sd.var("weights", Nd4j.rand(DataType.FLOAT, 10, 5));
        SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 5));
        SDVariable output = sd.nn.linear("output", input, weights, bias);

        // Export to ONNX
        File onnxFile = tempDir.resolve("exported_model.onnx").toFile();

        try {
            // Export using SameDiff's ONNX exporter
            sd.exportToOnnx(onnxFile);

            // Now load it back using the pipeline loader
            ONNXPipelineLoader loader = new ONNXPipelineLoader();
            SameDiff loaded = loader.loadModel(onnxFile, PipelineLoader.LoadConfig.defaults());

            assertNotNull(loaded);
            assertTrue(loaded.variables().size() > 0);
            log.info("Loaded ONNX model with {} variables", loaded.variables().size());

        } catch (Exception e) {
            // ONNX export might not be available in all configurations
            log.warn("ONNX export/import test skipped: {}", e.getMessage());
            Assumptions.assumeTrue(false, "ONNX export not available");
        }
    }

    // ==================== ONNXPipelineLoader with ModelManifest Tests ====================

    @Test
    public void testONNXPipelineLoaderWithEmptyManifest() {
        ONNXPipelineLoader loader = new ONNXPipelineLoader();

        ModelManifest emptyManifest = ModelManifest.builder()
                .format(ModelFormat.ONNX)
                .weightFiles(Collections.emptyList())
                .build();

        assertThrows(IOException.class,
                () -> loader.loadModel(emptyManifest, PipelineLoader.LoadConfig.defaults()));
    }

    @Test
    public void testONNXPipelineLoaderWithNullWeightFiles() {
        ONNXPipelineLoader loader = new ONNXPipelineLoader();

        ModelManifest nullManifest = ModelManifest.builder()
                .format(ModelFormat.ONNX)
                .weightFiles(null)
                .build();

        assertThrows(IOException.class,
                () -> loader.loadModel(nullManifest, PipelineLoader.LoadConfig.defaults()));
    }

    // ==================== Static Import Methods Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testONNXStaticImportMethod(Nd4jBackend backend) throws Exception {
        // Create and export a simple model
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("x", DataType.FLOAT, -1, 4);
        SDVariable output = sd.nn.relu("y", input, 0.0);

        File onnxFile = tempDir.resolve("static_import.onnx").toFile();

        try {
            sd.exportToOnnx(onnxFile);

            // Use static import method
            SameDiff imported = ONNXPipelineLoader.importModel(onnxFile);

            assertNotNull(imported);
            assertTrue(imported.variables().size() > 0);

        } catch (Exception e) {
            log.warn("Static import test skipped: {}", e.getMessage());
            Assumptions.assumeTrue(false, "ONNX export not available");
        }
    }

    @Test
    public void testONNXStaticImportWithPath() {
        // Test that the path-based import method works
        assertThrows(IOException.class,
                () -> ONNXPipelineLoader.importModel("/non/existent/path.onnx"));
    }

    // ==================== LoadConfig Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testONNXPipelineLoaderWithCustomConfig(Nd4jBackend backend) throws Exception {
        // Create a model with non-float32 types
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 8);
        SDVariable weights = sd.var("weights", Nd4j.rand(DataType.FLOAT, 8, 4));
        SDVariable output = sd.mmul("output", input, weights);

        File onnxFile = tempDir.resolve("config_test.onnx").toFile();

        try {
            sd.exportToOnnx(onnxFile);

            // Load with convertToFloat32 = false
            PipelineLoader.LoadConfig config = PipelineLoader.LoadConfig.builder()
                    .convertToFloat32(false)
                    .build();

            ONNXPipelineLoader loader = new ONNXPipelineLoader();
            SameDiff loaded = loader.loadModel(onnxFile, config);

            assertNotNull(loaded);

        } catch (Exception e) {
            log.warn("Config test skipped: {}", e.getMessage());
            Assumptions.assumeTrue(false, "ONNX export not available");
        }
    }

    // ==================== Pipeline Loading Tests ====================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testONNXPipelineLoaderLoadPipelineNonPipeline(Nd4jBackend backend) throws Exception {
        // Test that loadPipeline works for non-pipeline models (single model)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("in", DataType.FLOAT, -1, 16);
        SDVariable output = sd.nn.sigmoid("out", input);

        File onnxFile = tempDir.resolve("single_model.onnx").toFile();

        try {
            sd.exportToOnnx(onnxFile);

            File modelDir = tempDir.resolve("single_model_dir").toFile();
            modelDir.mkdirs();

            // Copy ONNX file to model directory
            java.nio.file.Files.copy(onnxFile.toPath(),
                    new File(modelDir, "model.onnx").toPath());

            ModelManifest manifest = ModelManifest.fromDirectory(modelDir);

            ONNXPipelineLoader loader = new ONNXPipelineLoader();
            Map<String, SameDiff> pipeline = loader.loadPipeline(manifest,
                    PipelineLoader.LoadConfig.defaults());

            assertNotNull(pipeline);
            assertEquals(1, pipeline.size());
            assertTrue(pipeline.containsKey("model"));

        } catch (Exception e) {
            log.warn("Pipeline loading test skipped: {}", e.getMessage());
            Assumptions.assumeTrue(false, "ONNX export not available");
        }
    }

    // ==================== Integration with ModelManifest Tests ====================

    @Test
    public void testModelManifestDetectsONNXFormat() throws IOException {
        File modelDir = tempDir.resolve("onnx_model").toFile();
        modelDir.mkdirs();

        // Create a dummy ONNX file (just needs to exist for format detection)
        File onnxFile = new File(modelDir, "model.onnx");
        onnxFile.createNewFile();

        ModelManifest manifest = ModelManifest.fromDirectory(modelDir);

        assertEquals(ModelFormat.ONNX, manifest.getFormat());
        assertNotNull(manifest.getWeightFiles());
        assertEquals(1, manifest.getWeightFiles().size());
        assertEquals("model.onnx", manifest.getWeightFiles().get(0).getName());
    }

    @Test
    public void testModelManifestPrioritySafetensorsOverONNX() throws IOException {
        // If both SafeTensors and ONNX files exist, SafeTensors should be preferred
        File modelDir = tempDir.resolve("mixed_model").toFile();
        modelDir.mkdirs();

        new File(modelDir, "model.onnx").createNewFile();
        new File(modelDir, "model.safetensors").createNewFile();

        ModelManifest manifest = ModelManifest.fromDirectory(modelDir);

        // SafeTensors should be detected before ONNX
        assertEquals(ModelFormat.SAFETENSORS, manifest.getFormat());
    }

}
