package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Compare SameDiff ONNX execution vs Python ONNX Runtime reference outputs.
 *
 * Pre-requisite: run the Python script that generates /tmp/ref_*.npy files.
 * Each test loads one ONNX model, feeds it the same input as ONNX Runtime,
 * and compares the output element-wise.
 */
@Slf4j
@NativeTag
@Tag(TagNames.FILE_IO)
@Tag("vlm")
public class TestVLMOnnxReference {

    private static final String MODEL_DIR = System.getProperty("vlm.model.cache.dir",
            System.getProperty("user.home") + "/.cache/dl4j-vlm-models");

    @Test
    @DisplayName("embed_tokens: SameDiff vs ONNX Runtime")
    public void testEmbedTokens() throws Exception {
        log.info("=== embed_tokens comparison ===");

        File refInput = new File("/tmp/ref_embed_input.npy");
        File refOutput = new File("/tmp/ref_embed_output.npy");
        assertTrue(refInput.exists(), "Run Python reference script first");
        assertTrue(refOutput.exists(), "Run Python reference script first");

        INDArray inputIds = Nd4j.createFromNpyFile(refInput);
        INDArray expectedOutput = Nd4j.createFromNpyFile(refOutput);
        log.info("Input: shape={}, values={}", Arrays.toString(inputIds.shape()), inputIds);
        log.info("Expected output: shape={}, min={}, max={}, mean={}",
                Arrays.toString(expectedOutput.shape()),
                expectedOutput.minNumber(), expectedOutput.maxNumber(), expectedOutput.meanNumber());

        // Import and run model
        String modelPath = MODEL_DIR + "/smoldocling-embed-tokens.onnx";
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff model = importer.runImport(modelPath, Map.of(), false, false);
        log.info("Model inputs: {}, outputs: {}", model.inputs(), model.outputs());

        INDArray inputLong = inputIds.castTo(DataType.LONG);
        String inputName = model.inputs().get(0);
        Map<String, INDArray> result = model.output(Map.of(inputName, inputLong),
                model.outputs().toArray(new String[0]));

        INDArray actual = null;
        for (var entry : result.entrySet()) {
            actual = entry.getValue();
            log.info("SameDiff output '{}': shape={}, min={}, max={}, mean={}",
                    entry.getKey(), Arrays.toString(actual.shape()),
                    actual.minNumber(), actual.maxNumber(), actual.meanNumber());
        }

        assertNotNull(actual);
        compareOutputs("embed_tokens", expectedOutput, actual);
    }

    @Test
    @DisplayName("vision encoder: SameDiff vs ONNX Runtime")
    public void testVisionEncoder() throws Exception {
        log.info("=== vision encoder comparison ===");

        File refPixels = new File("/tmp/ref_vision_input_pixels.npy");
        File refMask = new File("/tmp/ref_vision_input_mask.npy");
        File refOutput = new File("/tmp/ref_vision_output.npy");
        assertTrue(refPixels.exists(), "Run Python reference script first");

        INDArray pixelValues = Nd4j.createFromNpyFile(refPixels);
        INDArray mask = Nd4j.createFromNpyFile(refMask);
        INDArray expectedOutput = Nd4j.createFromNpyFile(refOutput);
        log.info("Pixel values: shape={}, min={}, max={}, mean={}",
                Arrays.toString(pixelValues.shape()),
                pixelValues.minNumber(), pixelValues.maxNumber(), pixelValues.meanNumber());
        log.info("Expected output: shape={}, min={}, max={}, mean={}",
                Arrays.toString(expectedOutput.shape()),
                expectedOutput.minNumber(), expectedOutput.maxNumber(), expectedOutput.meanNumber());

        // Import and run model
        String modelPath = MODEL_DIR + "/smoldocling-vision-encoder.onnx";
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff model = importer.runImport(modelPath, Map.of(), false, false);
        log.info("Model inputs: {}, outputs: {}", model.inputs(), model.outputs());

        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("pixel_values", pixelValues);
        inputs.put("pixel_attention_mask", mask);

        // Check intermediate outputs at key points to find where divergence starts
        String[] intermediateNodes = {
                "/GatherND_output_0",
                "/vision_model/embeddings/patch_embedding/Conv_output_0",
                // Embedding pipeline: conv reshape and transpose
                "/vision_model/embeddings/Reshape_output_0",
                "/vision_model/embeddings/Transpose_output_0",
                // Position ID computation - trace the full chain
                "/vision_model/embeddings/Reshape_3_output_0",
                "/vision_model/embeddings/NonZero_output_0",
                "/vision_model/embeddings/Transpose_1_output_0",
                "/vision_model/embeddings/Split_output_0",
                "/vision_model/embeddings/Split_output_1",
                "/vision_model/embeddings/Squeeze_output_0",
                "/vision_model/embeddings/Squeeze_1_output_0",
                "/vision_model/embeddings/Mul_6_output_0",
                "/vision_model/embeddings/Add_1_output_0",
                // Cast_2 (top-level boolean mask) and grid width computation
                "/Cast_2_output_0",
                "/ReduceSum_1_output_0",
                "/vision_model/embeddings/Gather_3_output_0",
                "/vision_model/embeddings/Cast_4_output_0",
                "/vision_model/embeddings/ReduceSum_output_0",
                // Div_2 inputs: Cast_8 / Cast_9
                "/vision_model/embeddings/Cast_9_output_0",
                // Row index computation: Div_2 → Less(Div_2, Expand_2) → Not → Cast_12 → ReduceSum_2 → Unsqueeze_7 → Mul_5
                "/vision_model/embeddings/Div_2_output_0",
                "/vision_model/embeddings/Expand_2_output_0",
                "/vision_model/embeddings/Less_output_0",
                "/vision_model/embeddings/Not_output_0",
                "/vision_model/embeddings/Cast_12_output_0",
                "/vision_model/embeddings/ReduceSum_2_output_0",
                "/vision_model/embeddings/Unsqueeze_7_output_0",
                // Add path (broadcast position grid): Mul_5 + Unsqueeze_8 → Add → Flatten
                "/vision_model/embeddings/Mul_5_output_0",
                "/vision_model/embeddings/Unsqueeze_8_output_0",
                "/vision_model/embeddings/Add_output_0",
                "/vision_model/embeddings/Flatten_output_0",
                // ScatterND path: Reshape_6, Concat_8 → ScatterND → Cast_14
                "/vision_model/embeddings/Reshape_6_output_0",
                "/vision_model/embeddings/Concat_8_output_0",
                "/vision_model/embeddings/Cast_14_output_0",
                // Position embeddings and final embedding output
                "/vision_model/embeddings/position_embedding/Gather_output_0",
                "/vision_model/embeddings/Add_3_output_0",
                // Layer 0 layer_norm1 - just check input and output
                "/vision_model/encoder/layers.0/layer_norm1/Sub_output_0",
                "/vision_model/encoder/layers.0/layer_norm1/Add_1_output_0",
        };

        // Collect outputs: final + intermediates that exist in the graph
        java.util.List<String> outputNames = new java.util.ArrayList<>(model.outputs());
        for (String node : intermediateNodes) {
            if (model.hasVariable(node)) {
                outputNames.add(node);
            } else {
                log.warn("Intermediate node not found: {}", node);
            }
        }

        Map<String, INDArray> result = model.output(inputs, outputNames.toArray(new String[0]));

        // Compare intermediates against Python references
        for (String node : intermediateNodes) {
            INDArray sdOut = result.get(node);
            if (sdOut == null) continue;
            String safeName = node.replace('/', '_').replace('.', '_');
            // Try ref_intermediate_, ref_early_, and ref_emb_ prefixes
            File refFile = new File("/tmp/ref_intermediate_" + safeName + ".npy");
            if (!refFile.exists()) {
                refFile = new File("/tmp/ref_early_" + safeName + ".npy");
            }
            if (!refFile.exists()) {
                refFile = new File("/tmp/ref_emb_" + safeName + ".npy");
            }
            if (refFile.exists()) {
                INDArray pyOut = Nd4j.createFromNpyFile(refFile);
                compareOutputs(node, pyOut, sdOut);
            } else {
                log.info("  {} (no ref): shape={}, min={}, max={}, mean={}",
                        node, Arrays.toString(sdOut.shape()),
                        sdOut.minNumber(), sdOut.maxNumber(), sdOut.meanNumber());
            }
        }

        INDArray actual = null;
        for (String outName : model.outputs()) {
            actual = result.get(outName);
            if (actual != null) {
                log.info("SameDiff output '{}': shape={}, min={}, max={}, mean={}",
                        outName, Arrays.toString(actual.shape()),
                        actual.minNumber(), actual.maxNumber(), actual.meanNumber());
            }
        }

        assertNotNull(actual);
        compareOutputs("vision_encoder", expectedOutput, actual);
    }

    @Test
    @DisplayName("decoder: SameDiff vs ONNX Runtime")
    public void testDecoder() throws Exception {
        log.info("=== decoder comparison ===");

        File refEmbeds = new File("/tmp/ref_decoder_input_embeds.npy");
        File refLogits = new File("/tmp/ref_decoder_output_logits.npy");
        assertTrue(refEmbeds.exists(), "Run Python reference script first");

        INDArray inputEmbeds = Nd4j.createFromNpyFile(refEmbeds).castTo(DataType.FLOAT);
        INDArray expectedLogits = Nd4j.createFromNpyFile(refLogits);
        long seqLen = inputEmbeds.shape()[1];
        log.info("Input embeds: shape={}, min={}, max={}",
                Arrays.toString(inputEmbeds.shape()),
                inputEmbeds.minNumber(), inputEmbeds.maxNumber());
        log.info("Expected logits: shape={}, min={}, max={}",
                Arrays.toString(expectedLogits.shape()),
                expectedLogits.minNumber(), expectedLogits.maxNumber());

        // Import model
        String modelPath = MODEL_DIR + "/smoldocling-decoder.onnx";
        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff model = importer.runImport(modelPath, Map.of(), false, false);
        log.info("Model inputs: {}, outputs: {}", model.inputs(), model.outputs());

        // Find logits output
        String logitsName = null;
        for (String name : model.outputs()) {
            if (name.contains("logit")) {
                logitsName = name;
                break;
            }
        }
        if (logitsName == null) logitsName = model.outputs().get(0);

        // Build decoder inputs
        Map<String, INDArray> inputs = new HashMap<>();
        for (String name : model.inputs()) {
            if (name.equals("inputs_embeds")) {
                inputs.put(name, inputEmbeds);
            } else if (name.equals("attention_mask")) {
                inputs.put(name, Nd4j.ones(DataType.LONG, 1, seqLen));
            } else if (name.equals("position_ids")) {
                inputs.put(name, Nd4j.arange(seqLen).reshape(1, seqLen).castTo(DataType.LONG));
            } else if (name.startsWith("past_key_values.")) {
                inputs.put(name, Nd4j.zeros(DataType.FLOAT, 1, 3, 0, 64));
            }
        }

        Map<String, INDArray> result = model.output(inputs, new String[]{logitsName});
        INDArray actualLogits = result.get(logitsName);
        log.info("SameDiff logits: shape={}, min={}, max={}",
                Arrays.toString(actualLogits.shape()),
                actualLogits.minNumber(), actualLogits.maxNumber());

        // Compare last position argmax
        INDArray expectedLast = expectedLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        INDArray actualLast = actualLogits.get(NDArrayIndex.point(0),
                NDArrayIndex.point(seqLen - 1), NDArrayIndex.all());
        int expectedArgmax = expectedLast.argMax().getInt(0);
        int actualArgmax = actualLast.argMax().getInt(0);
        log.info("Last position argmax: expected={}, actual={}", expectedArgmax, actualArgmax);

        compareOutputs("decoder_logits", expectedLogits, actualLogits);
    }

    @Test
    @DisplayName("direct conv2d: nd4j op vs ONNX Runtime")
    public void testDirectConv() throws Exception {
        log.info("=== direct conv2d via nd4j op ===");

        INDArray input = Nd4j.createFromNpyFile(new File("/tmp/ref_intermediate__GatherND_output_0.npy")); // [1,3,512,512] NCHW
        INDArray weight = Nd4j.createFromNpyFile(new File("/tmp/ref_conv_weight.npy")); // [768,3,16,16] OIHW
        INDArray bias = Nd4j.createFromNpyFile(new File("/tmp/ref_conv_bias.npy"));     // [768]
        INDArray expected = Nd4j.createFromNpyFile(new File("/tmp/ref_conv_output.npy")); // [1,768,32,32] NCHW

        log.info("Input NCHW: {}", Arrays.toString(input.shape()));
        log.info("Weight OIHW: {}", Arrays.toString(weight.shape()));

        // Approach 1: Use conv2d with NCHW format and OIYX weights (no permutation)
        {
            log.info("--- Approach 1: NCHW input, OIYX weight ---");
            SameDiff sd = SameDiff.create();
            SDVariable sdInput = sd.var("input", input);
            SDVariable sdWeight = sd.var("weight", weight);
            SDVariable sdBias = sd.var("bias", bias);

            Conv2DConfig config = Conv2DConfig.builder()
                    .kH(16).kW(16).sH(16).sW(16)
                    .pH(0).pW(0).dH(1).dW(1)
                    .dataFormat("NCHW")
                    .weightsFormat(WeightsFormat.OIYX)
                    .paddingMode(PaddingMode.VALID)
                    .build();
            SDVariable conv = sd.cnn().conv2d(sdInput, sdWeight, sdBias, config);

            Map<String, INDArray> result = sd.output(Map.of(), conv.name());
            INDArray actual = result.get(conv.name());
            log.info("Output: shape={}, min={}, max={}", Arrays.toString(actual.shape()),
                    actual.minNumber(), actual.maxNumber());
            compareOutputs("direct_conv_NCHW_OIYX", expected, actual);
        }

        // Approach 2: Reproduce what the ONNX import does (permute to NHWC+YXIO)
        {
            log.info("--- Approach 2: NHWC input, YXIO weight (ONNX import style) ---");
            // Permute input NCHW [0,1,2,3] -> NHWC [0,2,3,1]
            INDArray inputNHWC = input.permute(0, 2, 3, 1).dup();
            // Permute weight OIHW [0,1,2,3] -> YXIO [2,3,1,0]
            INDArray weightYXIO = weight.permute(2, 3, 1, 0).dup();

            log.info("Input NHWC: {}", Arrays.toString(inputNHWC.shape()));
            log.info("Weight YXIO: {}", Arrays.toString(weightYXIO.shape()));

            SameDiff sd = SameDiff.create();
            SDVariable sdInput = sd.var("input", inputNHWC);
            SDVariable sdWeight = sd.var("weight", weightYXIO);

            Conv2DConfig config2 = Conv2DConfig.builder()
                    .kH(16).kW(16).sH(16).sW(16)
                    .pH(0).pW(0).dH(1).dW(1)
                    .dataFormat("NHWC")
                    .weightsFormat(WeightsFormat.YXIO)
                    .paddingMode(PaddingMode.VALID)
                    .build();
            SDVariable conv = sd.cnn().conv2d(sdInput, sdWeight, config2);
            // Add bias in NHWC (last dim = channels)
            SDVariable sdBias = sd.var("bias", bias);
            SDVariable withBias = conv.add(sdBias);
            // Permute back NHWC -> NCHW [0,3,1,2]
            SDVariable outNCHW = sd.permute(withBias, 0, 3, 1, 2);

            Map<String, INDArray> result = sd.output(Map.of(), outNCHW.name());
            INDArray actual = result.get(outNCHW.name());
            log.info("Output: shape={}, min={}, max={}", Arrays.toString(actual.shape()),
                    actual.minNumber(), actual.maxNumber());
            compareOutputs("direct_conv_NHWC_YXIO", expected, actual);
        }
    }

    @Test
    @DisplayName("isolated conv2d: SameDiff vs ONNX Runtime")
    public void testIsolatedConv() throws Exception {
        log.info("=== isolated conv2d comparison ===");

        File refInput = new File("/tmp/ref_intermediate__GatherND_output_0.npy");
        File refOutput = new File("/tmp/ref_conv_output.npy");
        File convModel = new File("/tmp/test_conv_only.onnx");
        assertTrue(refInput.exists(), "Run Python reference script first");
        assertTrue(refOutput.exists(), "Run Python reference script first");
        assertTrue(convModel.exists(), "Run Python reference script first");

        INDArray input = Nd4j.createFromNpyFile(refInput);
        INDArray expected = Nd4j.createFromNpyFile(refOutput);
        log.info("Input: shape={}, min={}, max={}", Arrays.toString(input.shape()),
                input.minNumber(), input.maxNumber());
        log.info("Expected: shape={}, min={}, max={}", Arrays.toString(expected.shape()),
                expected.minNumber(), expected.maxNumber());

        OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
        SameDiff model = importer.runImport(convModel.getAbsolutePath(), Map.of(), false, false);
        log.info("Conv model inputs: {}, outputs: {}", model.inputs(), model.outputs());

        // Check loaded weights
        INDArray refWeight = Nd4j.createFromNpyFile(new File("/tmp/ref_conv_weight.npy"));
        INDArray refBias = Nd4j.createFromNpyFile(new File("/tmp/ref_conv_bias.npy"));
        for (String varName : model.getVariables().keySet()) {
            var sdVar = model.getVariable(varName);
            if (sdVar.getArr() != null && sdVar.getArr().length() > 1) {
                INDArray arr = sdVar.getArr();
                log.info("Variable '{}': type={}, shape={}, dtype={}, min={}, max={}",
                        varName, sdVar.getVariableType(), Arrays.toString(arr.shape()),
                        arr.dataType(), arr.minNumber(), arr.maxNumber());
                if (arr.shape().length == 4 && arr.size(0) == 768) {
                    // This is likely the conv weight
                    INDArray diff = arr.castTo(DataType.FLOAT).sub(refWeight);
                    log.info("Weight diff vs Python ref: maxAbsDiff={}, L2={}",
                            Nd4j.math().abs(diff).maxNumber(), diff.norm2Number());
                    // Check first kernel values
                    float[] sdFirst = arr.reshape(-1).get(NDArrayIndex.interval(0, 10)).toFloatVector();
                    float[] pyFirst = refWeight.reshape(-1).get(NDArrayIndex.interval(0, 10)).toFloatVector();
                    log.info("Weight first 10 SD:  {}", Arrays.toString(sdFirst));
                    log.info("Weight first 10 Py:  {}", Arrays.toString(pyFirst));
                }
            }
        }

        String inputName = model.inputs().get(0);
        Map<String, INDArray> result = model.output(Map.of(inputName, input),
                model.outputs().toArray(new String[0]));

        INDArray actual = null;
        for (var entry : result.entrySet()) {
            actual = entry.getValue();
            log.info("SameDiff output '{}': shape={}, min={}, max={}",
                    entry.getKey(), Arrays.toString(actual.shape()),
                    actual.minNumber(), actual.maxNumber());
        }

        assertNotNull(actual);
        compareOutputs("isolated_conv", expected, actual);
    }

    @Test
    @DisplayName("conv2d diagnostic: small known input + manual verification")
    public void testConv2dDiagnostic() throws Exception {
        log.info("=== conv2d diagnostic ===");

        INDArray input = Nd4j.create(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4});  // NCHW
        INDArray weight = Nd4j.ones(DataType.FLOAT, 1, 1, 2, 2);

        log.info("Input ordering: {}, shape: {}, stride: {}", input.ordering(),
                Arrays.toString(input.shape()), Arrays.toString(input.stride()));
        log.info("Input data: {}", Arrays.toString(input.dup().reshape(-1).toFloatVector()));

        Conv2DConfig config = Conv2DConfig.builder()
                .kH(2).kW(2).sH(2).sW(2)
                .pH(0).pW(0).dH(1).dW(1)
                .dataFormat("NCHW")
                .weightsFormat(WeightsFormat.OIYX)
                .paddingMode(PaddingMode.VALID)
                .build();

        // A: SameDiff
        {
            SameDiff sd = SameDiff.create();
            SDVariable sdIn = sd.var("in", input);
            SDVariable sdW = sd.var("w", weight);
            SDVariable out = sd.cnn().conv2d(sdIn, sdW, config);
            Map<String, INDArray> result = sd.output(Map.of(), out.name());
            INDArray actual = result.get(out.name());
            log.info("SameDiff output: {}", Arrays.toString(actual.dup().reshape(-1).toFloatVector()));
        }

        // B: Direct op (bypass SameDiff)
        {
            INDArray output = Nd4j.create(DataType.FLOAT, 1, 1, 2, 2);
            org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D op =
                    new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(
                            input, weight, null, output, config);
            Nd4j.exec(op);
            log.info("Direct op output: {}", Arrays.toString(output.dup().reshape(-1).toFloatVector()));
        }

        // C: Manual im2col + matmul
        {
            INDArray col = Nd4j.create(DataType.FLOAT, 4, 4);
            for (int oh = 0; oh < 2; oh++) {
                for (int ow = 0; ow < 2; ow++) {
                    int patchIdx = oh * 2 + ow;
                    int colIdx = 0;
                    for (int kh = 0; kh < 2; kh++) {
                        for (int kw = 0; kw < 2; kw++) {
                            col.putScalar(patchIdx, colIdx, input.getDouble(0, 0, oh * 2 + kh, ow * 2 + kw));
                            colIdx++;
                        }
                    }
                }
            }
            INDArray wFlat = weight.reshape(1, 4).transpose();
            INDArray mmResult = col.mmul(wFlat);
            log.info("Manual matmul output: {}", Arrays.toString(mmResult.dup().reshape(-1).toFloatVector()));
        }

        log.info("Expected: [14.0, 22.0, 46.0, 54.0]");

        // Test 2: stride 1 (overlapping) to see if stride is the issue
        {
            log.info("--- Test 2: same input, stride 1, kernel 2 ---");
            Conv2DConfig cfg2 = Conv2DConfig.builder()
                    .kH(2).kW(2).sH(1).sW(1)
                    .pH(0).pW(0).dH(1).dW(1)
                    .dataFormat("NCHW")
                    .weightsFormat(WeightsFormat.OIYX)
                    .paddingMode(PaddingMode.VALID)
                    .build();
            INDArray out2 = Nd4j.create(DataType.FLOAT, 1, 1, 3, 3);
            org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D op2 =
                    new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(
                            input, weight, null, out2, cfg2);
            Nd4j.exec(op2);
            log.info("Stride-1 output: {}", Arrays.toString(out2.dup().reshape(-1).toFloatVector()));
            // Expected: [14, 18, 22, 30, 34, 38, 46, 50, 54]
            log.info("Expected stride-1: [14, 18, 22, 30, 34, 38, 46, 50, 54]");
        }

        // Test 3: multi-channel
        {
            log.info("--- Test 3: 3ch->2ch, 4x4 kernel, stride 4, 8x8 input ---");
            INDArray input3 = Nd4j.linspace(0, 191, 192, DataType.FLOAT).reshape(1, 3, 8, 8);
            INDArray weight3 = Nd4j.linspace(-0.1, 0.086, 96, DataType.FLOAT).reshape(2, 3, 4, 4);

            Conv2DConfig cfg3 = Conv2DConfig.builder()
                    .kH(4).kW(4).sH(4).sW(4)
                    .pH(0).pW(0).dH(1).dW(1)
                    .dataFormat("NCHW")
                    .weightsFormat(WeightsFormat.OIYX)
                    .paddingMode(PaddingMode.VALID)
                    .build();

            INDArray out3 = Nd4j.create(DataType.FLOAT, 1, 2, 2, 2);
            org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D op3 =
                    new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(
                            input3, weight3, null, out3, cfg3);
            Nd4j.exec(op3);
            log.info("Multi-ch output: {}", Arrays.toString(out3.dup().reshape(-1).toFloatVector()));

            double manual00 = 0;
            for (int ic = 0; ic < 3; ic++) {
                for (int kh = 0; kh < 4; kh++) {
                    for (int kw = 0; kw < 4; kw++) {
                        manual00 += input3.getDouble(0, ic, kh, kw) * weight3.getDouble(0, ic, kh, kw);
                    }
                }
            }
            log.info("Manual [0,0,0,0]={}, op={}", manual00, out3.getDouble(0, 0, 0, 0));
        }
    }

    private void compareOutputs(String name, INDArray expected, INDArray actual) {
        log.info("--- {} comparison ---", name);
        log.info("Expected: shape={}, dtype={}", Arrays.toString(expected.shape()), expected.dataType());
        log.info("Actual:   shape={}, dtype={}", Arrays.toString(actual.shape()), actual.dataType());

        // Cast to same type for comparison
        INDArray exp = expected.castTo(DataType.FLOAT);
        INDArray act = actual.castTo(DataType.FLOAT);

        INDArray diff = exp.sub(act);
        double maxAbsDiff = Nd4j.math().abs(diff).maxNumber().doubleValue();
        double meanAbsDiff = Nd4j.math().abs(diff).meanNumber().doubleValue();
        double l2 = diff.norm2Number().doubleValue();
        double relErr = l2 / (exp.norm2Number().doubleValue() + 1e-8);

        log.info("maxAbsDiff={}, meanAbsDiff={}, L2={}, relativeL2={}", maxAbsDiff, meanAbsDiff, l2, relErr);

        // Check first 10 values
        float[] expFirst = exp.dup().reshape(-1).get(NDArrayIndex.interval(0, Math.min(10, exp.length()))).toFloatVector();
        float[] actFirst = act.dup().reshape(-1).get(NDArrayIndex.interval(0, Math.min(10, act.length()))).toFloatVector();
        log.info("First 10 expected: {}", Arrays.toString(expFirst));
        log.info("First 10 actual:   {}", Arrays.toString(actFirst));
    }
}
