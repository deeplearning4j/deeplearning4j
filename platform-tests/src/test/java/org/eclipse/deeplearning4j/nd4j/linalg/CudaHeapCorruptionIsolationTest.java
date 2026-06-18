package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolates the heap corruption that crashes the VLM test.
 *
 * The VLM test crashes with "corrupted size vs. prev_size" during
 * CudaMemoryPool init. CudaMemoryPool works fine in isolation
 * (see CudaMemoryPoolTest). Something during VLM class loading
 * or static init corrupts the heap before CudaMemoryPool detects it.
 *
 * This test loads VLM-related classes one by one to find which one
 * triggers the corruption.
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class CudaHeapCorruptionIsolationTest {

    @Test
    @Order(1)
    @DisplayName("Phase 1: Basic Nd4j init")
    public void phase1_basicInit() {
        log.info("Phase 1: Basic Nd4j init");
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 10);
        assertNotNull(arr);
        arr.close();
        log.info("Phase 1 PASSED");
    }

    @Test
    @Order(2)
    @DisplayName("Phase 2: Load SameDiff")
    public void phase2_loadSameDiff() throws Exception {
        log.info("Phase 2: Loading SameDiff");
        Class<?> cls = Class.forName("org.nd4j.autodiff.samediff.SameDiff");
        assertNotNull(cls);
        // Verify heap is still ok by doing an allocation
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024);
        arr.close();
        log.info("Phase 2 PASSED");
    }

    @Test
    @Order(3)
    @DisplayName("Phase 3: Load OnnxFrameworkImporter")
    public void phase3_loadOnnxImporter() throws Exception {
        log.info("Phase 3: Loading OnnxFrameworkImporter");
        Class<?> cls = Class.forName("org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter");
        assertNotNull(cls);
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024);
        arr.close();
        log.info("Phase 3 PASSED");
    }

    @Test
    @Order(4)
    @DisplayName("Phase 4: Load GGMLModelImport")
    public void phase4_loadGGML() throws Exception {
        log.info("Phase 4: Loading GGMLModelImport");
        Class<?> cls = Class.forName("org.nd4j.ggml.GGMLModelImport");
        assertNotNull(cls);
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024);
        arr.close();
        log.info("Phase 4 PASSED");
    }

    @Test
    @Order(5)
    @DisplayName("Phase 5: Load VLM preprocessing classes")
    public void phase5_loadVLMPreprocessing() throws Exception {
        log.info("Phase 5: Loading VLM preprocessing");
        Class.forName("org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor");
        Class.forName("org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler");
        Class.forName("org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig");
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024);
        arr.close();
        log.info("Phase 5 PASSED");
    }

    @Test
    @Order(6)
    @DisplayName("Phase 6: Load VLM model classes")
    public void phase6_loadVLMModel() throws Exception {
        log.info("Phase 6: Loading VLM model classes");
        Class.forName("org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils");
        Class.forName("org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger");
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024);
        arr.close();
        log.info("Phase 6 PASSED");
    }

    @Test
    @Order(7)
    @DisplayName("Phase 7: Load LLM generation classes")
    public void phase7_loadLLMGeneration() throws Exception {
        log.info("Phase 7: Loading LLM generation");
        Class.forName("org.eclipse.deeplearning4j.llm.generation.TextGenerator");
        Class.forName("org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer");
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024);
        arr.close();
        log.info("Phase 7 PASSED");
    }

    @Test
    @Order(8)
    @DisplayName("Phase 8: Create SameDiff instance and run simple graph")
    public void phase8_runSameDiff() {
        log.info("Phase 8: SameDiff graph execution");
        var sd = org.nd4j.autodiff.samediff.SameDiff.create();
        var input = sd.placeHolder("input", DataType.FLOAT, -1, 10);
        var output = sd.math().add("output", input, input);
        var result = sd.output(
                java.util.Collections.singletonMap("input", Nd4j.ones(DataType.FLOAT, 2, 10)),
                "output"
        );
        assertNotNull(result.get("output"));
        log.info("Phase 8 PASSED: result shape={}", result.get("output").shape());
    }

    @Test
    @Order(9)
    @DisplayName("Phase 9: Import small ONNX model (if available)")
    public void phase9_onnxImport() throws Exception {
        log.info("Phase 9: ONNX import test");
        // Try to import a small model to exercise the ONNX import + CUDA path
        String modelPath = System.getProperty("vlm.model.cache.dir",
                System.getProperty("user.home") + "/.cache/nd4j/models");
        java.io.File cacheDir = new java.io.File(modelPath);
        if (!cacheDir.exists()) {
            log.info("Phase 9 SKIPPED: no model cache directory");
            return;
        }

        // Just verify we can create an importer without crashing
        var importer = new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        assertNotNull(importer);
        log.info("Phase 9 PASSED: OnnxFrameworkImporter created");
    }
}
