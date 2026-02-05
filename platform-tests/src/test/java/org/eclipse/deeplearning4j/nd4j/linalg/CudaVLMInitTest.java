package org.eclipse.deeplearning4j.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
// Import EXACTLY the same classes as TestVLMModelImportPipeline
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.DecoderGraphFixer;
import org.eclipse.deeplearning4j.vlm.model.EmbeddingMerger;
import org.eclipse.deeplearning4j.vlm.model.VisionEncoderUtils;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImagePromptBuilder;
import org.eclipse.deeplearning4j.vlm.preprocessing.ImageTiler;
import org.eclipse.deeplearning4j.vlm.preprocessing.PreprocessorConfig;
import org.eclipse.deeplearning4j.vlm.preprocessing.VLMImagePreprocessor;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.eclipse.deeplearning4j.llm.generation.*;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.llm.tokenizer.Tokenizer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Reproduces the VLM test crash by importing the exact same classes.
 * If this crashes with "corrupted size vs. prev_size" the same way,
 * the bug is in class loading / static init. If it passes, the bug
 * is in the test method itself.
 */
@Slf4j
public class CudaVLMInitTest {

    @Test
    @DisplayName("VLM init - same imports as TestVLMModelImportPipeline")
    public void testVLMInitSameImports() {
        log.info("Starting VLM init test with same imports");
        log.info("Nd4j backend: {}", Nd4j.getBackend().getClass().getSimpleName());

        // Basic CUDA operation
        INDArray arr = Nd4j.zeros(DataType.FLOAT, 10, 10);
        assertNotNull(arr);
        log.info("Array created: shape={}", arr.shape());

        // Trigger some of the imported classes to ensure they're actually loaded
        log.info("VLMModelDownloader cache dir: {}", VLMModelDownloader.getCacheDir());
        log.info("PreprocessorConfig exists: {}", PreprocessorConfig.class.getSimpleName());

        // Do a trim to exercise CudaMemoryPool
        NativeOpsHolder.getInstance().getDeviceNativeOps().trimMemoryPool(0);

        arr.close();
        log.info("TEST PASSED - no heap corruption");
    }
}
