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
package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx;

import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnx.OnnxTensorUtils;
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Utility class for building small single-op ONNX models and comparing
 * outputs between ONNX Runtime (ground truth) and SameDiff (our import).
 */
@Slf4j
public class OnnxOpTestHelper {

    public static final int DEFAULT_OPSET = 13;
    public static final String DEFAULT_DOMAIN = "";  // ai.onnx

    /**
     * Build a minimal ONNX model with a single op node.
     */
    public static Onnx.ModelProto buildModel(
            String opType,
            List<InputSpec> inputs,
            List<OutputSpec> outputs,
            Map<String, Onnx.AttributeProto> attributes,
            int opsetVersion,
            String domain) {

        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_" + opType);

        // Build the single op node
        Onnx.NodeProto.Builder nodeBuilder = Onnx.NodeProto.newBuilder()
                .setOpType(opType)
                .setName(opType + "_0");

        if (domain != null && !domain.isEmpty()) {
            nodeBuilder.setDomain(domain);
        }

        // Add inputs - separate graph inputs from initializers
        for (InputSpec input : inputs) {
            nodeBuilder.addInput(input.name);
            if (input.initializer != null) {
                // This is a constant - add as initializer only, NOT as graph input.
                // In ONNX opset 13+, initializers not in graph inputs are true constants.
                graphBuilder.addInitializer(
                        OnnxTensorUtils.toTensorProto(input.initializer, input.name));
            } else {
                graphBuilder.addInput(makeValueInfo(input.name, input.dataType, input.shape));
            }
        }

        // Add outputs
        for (OutputSpec output : outputs) {
            nodeBuilder.addOutput(output.name);
            graphBuilder.addOutput(makeValueInfo(output.name, output.dataType, output.shape));
        }

        // Add attributes
        if (attributes != null) {
            for (Onnx.AttributeProto attr : attributes.values()) {
                nodeBuilder.addAttribute(attr);
            }
        }

        graphBuilder.addNode(nodeBuilder.build());

        // Build model
        Onnx.ModelProto.Builder modelBuilder = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder()
                        .setVersion(opsetVersion)
                        .setDomain(domain != null ? domain : "")
                        .build());

        // Always include the default ai.onnx opset if using a custom domain
        if (domain != null && !domain.isEmpty()) {
            modelBuilder.addOpsetImport(Onnx.OperatorSetIdProto.newBuilder()
                    .setVersion(DEFAULT_OPSET)
                    .setDomain("")
                    .build());
        }

        return modelBuilder.build();
    }

    /**
     * Build a model with default opset and domain.
     */
    public static Onnx.ModelProto buildModel(
            String opType,
            List<InputSpec> inputs,
            List<OutputSpec> outputs,
            Map<String, Onnx.AttributeProto> attributes) {
        return buildModel(opType, inputs, outputs, attributes, DEFAULT_OPSET, DEFAULT_DOMAIN);
    }

    /**
     * Save model to file.
     */
    public static File saveModel(Onnx.ModelProto model, Path directory) throws IOException {
        File modelFile = directory.resolve("model.onnx").toFile();
        Files.write(modelFile.toPath(), model.toByteArray());
        return modelFile;
    }

    /**
     * Create a ValueInfoProto for a graph input/output.
     */
    public static Onnx.ValueInfoProto makeValueInfo(String name, DataType dataType, long[] shape) {
        Onnx.TypeProto.Tensor.Builder tensorBuilder = Onnx.TypeProto.Tensor.newBuilder()
                .setElemType(dataTypeToOnnx(dataType));

        if (shape != null) {
            Onnx.TensorShapeProto.Builder shapeBuilder = Onnx.TensorShapeProto.newBuilder();
            for (long dim : shape) {
                shapeBuilder.addDim(
                        Onnx.TensorShapeProto.Dimension.newBuilder().setDimValue(dim).build());
            }
            tensorBuilder.setShape(shapeBuilder.build());
        }

        return Onnx.ValueInfoProto.newBuilder()
                .setName(name)
                .setType(Onnx.TypeProto.newBuilder()
                        .setTensorType(tensorBuilder.build())
                        .build())
                .build();
    }

    /**
     * Compare outputs with configurable tolerance.
     * Returns true if all outputs match within tolerance.
     */
    public static void compareOutputs(
            Map<String, INDArray> expected,
            Map<String, INDArray> actual,
            double maxRelError,
            double minAbsError) {

        for (Map.Entry<String, INDArray> entry : expected.entrySet()) {
            String name = entry.getKey();
            INDArray exp = entry.getValue();

            // SameDiff may use different output naming - try common patterns
            INDArray act = findMatchingOutput(name, actual);
            assertNotNull(act, "Missing output: " + name + ". Available: " + actual.keySet());

            // Cast to same type for comparison
            if (exp.dataType() != act.dataType()) {
                act = act.castTo(exp.dataType());
            }

            // Check shapes match
            assertArrayEquals(exp.shape(), act.shape(),
                    "Shape mismatch for output '" + name + "': expected " +
                            Arrays.toString(exp.shape()) + " but got " + Arrays.toString(act.shape()));

            // Element-wise comparison
            if (exp.dataType().isFPType()) {
                for (long i = 0; i < exp.length(); i++) {
                    double expVal = exp.getDouble(i);
                    double actVal = act.getDouble(i);
                    double diff = Math.abs(expVal - actVal);
                    double relDiff = expVal == 0.0 ? diff : diff / Math.max(Math.abs(expVal), 1e-20);

                    if (diff > minAbsError && relDiff > maxRelError) {
                        fail(String.format(
                                "Output '%s' mismatch at index %d: expected %f but got %f (abs diff: %e, rel diff: %e)",
                                name, i, expVal, actVal, diff, relDiff));
                    }
                }
            } else {
                assertEquals(exp, act, "Output '" + name + "' mismatch");
            }
        }
    }

    /**
     * Find matching output by name, trying common SameDiff naming patterns.
     */
    private static INDArray findMatchingOutput(String onnxName, Map<String, INDArray> outputs) {
        // Direct match
        if (outputs.containsKey(onnxName)) {
            return outputs.get(onnxName);
        }
        // SameDiff may append ":0" or use different naming
        for (Map.Entry<String, INDArray> entry : outputs.entrySet()) {
            if (entry.getKey().endsWith(onnxName) || onnxName.endsWith(entry.getKey())) {
                return entry.getValue();
            }
        }
        // If only one output, just return it
        if (outputs.size() == 1) {
            return outputs.values().iterator().next();
        }
        return null;
    }

    /**
     * Execute an op test end-to-end: build model, run with ORT and SameDiff, compare.
     */
    public static OpTestResult executeOpTest(
            String opType,
            List<InputSpec> inputs,
            List<OutputSpec> outputs,
            Map<String, Onnx.AttributeProto> attributes,
            Path tempDir,
            double maxRelError,
            double minAbsError) throws Exception {
        return executeOpTest(opType, inputs, outputs, attributes, tempDir,
                maxRelError, minAbsError, DEFAULT_OPSET, DEFAULT_DOMAIN);
    }

    /**
     * Execute an op test end-to-end with explicit opset and domain.
     */
    public static OpTestResult executeOpTest(
            String opType,
            List<InputSpec> inputs,
            List<OutputSpec> outputs,
            Map<String, Onnx.AttributeProto> attributes,
            Path tempDir,
            double maxRelError,
            double minAbsError,
            int opsetVersion,
            String domain) throws Exception {

        OpTestResult result = new OpTestResult(opType);

        // Build and save model
        Onnx.ModelProto model = buildModel(opType, inputs, outputs, attributes, opsetVersion, domain);
        File modelFile = saveModel(model, tempDir);
        result.modelFile = modelFile;

        // Prepare input map (only non-initializer inputs)
        // Sync all arrays to host for ORT (which reads from CPU memory).
        // On CUDA, Nd4j.rand() generates on GPU and the host buffer is stale.
        Nd4j.getExecutioner().commit();
        Map<String, INDArray> inputMap = new LinkedHashMap<>();
        for (InputSpec input : inputs) {
            if (input.initializer == null) {
                inputMap.put(input.name, input.data);
            }
        }

        // Run with ONNX Runtime
        try (OnnxRuntimeRunner runner = OnnxRuntimeRunner.builder()
                .modelUri(modelFile.getAbsolutePath())
                .build()) {
            result.onnxRuntimeOutputs = runner.exec(inputMap);
        } catch (Exception e) {
            result.onnxRuntimeError = e;
            log.warn("ONNX Runtime failed for {}: {}", opType, e.getMessage());
            return result;
        }

        // Run with SameDiff import
        try {
            OnnxFrameworkImporter importer = new OnnxFrameworkImporter();
            SameDiff sd = importer.runImport(modelFile.getAbsolutePath(), inputMap, false, false);
            result.sameDiffOutputs = sd.outputAll(inputMap);
        } catch (Exception e) {
            result.sameDiffError = e;
            log.warn("SameDiff import/exec failed for {}: {}", opType, e.getMessage());
            return result;
        }

        // Log output keys for debugging
        log.debug("Op {} ORT outputs: {}", opType, result.onnxRuntimeOutputs.keySet());
        log.debug("Op {} SD outputs: {}", opType, result.sameDiffOutputs.keySet());

        // Compare outputs
        try {
            compareOutputs(result.onnxRuntimeOutputs, result.sameDiffOutputs, maxRelError, minAbsError);
            result.passed = true;
        } catch (Throwable e) {
            result.comparisonError = e;
            log.warn("Output comparison failed for {}: {}", opType, e.getMessage());
        }

        return result;
    }

    // --- Attribute builder helpers ---

    public static Onnx.AttributeProto intAttr(String name, long value) {
        return Onnx.AttributeProto.newBuilder()
                .setName(name)
                .setType(Onnx.AttributeProto.AttributeType.INT)
                .setI(value)
                .build();
    }

    public static Onnx.AttributeProto floatAttr(String name, float value) {
        return Onnx.AttributeProto.newBuilder()
                .setName(name)
                .setType(Onnx.AttributeProto.AttributeType.FLOAT)
                .setF(value)
                .build();
    }

    public static Onnx.AttributeProto stringAttr(String name, String value) {
        return Onnx.AttributeProto.newBuilder()
                .setName(name)
                .setType(Onnx.AttributeProto.AttributeType.STRING)
                .setS(org.nd4j.shade.protobuf.ByteString.copyFromUtf8(value))
                .build();
    }

    public static Onnx.AttributeProto intsAttr(String name, long... values) {
        Onnx.AttributeProto.Builder builder = Onnx.AttributeProto.newBuilder()
                .setName(name)
                .setType(Onnx.AttributeProto.AttributeType.INTS);
        for (long v : values) {
            builder.addInts(v);
        }
        return builder.build();
    }

    public static Onnx.AttributeProto floatsAttr(String name, float... values) {
        Onnx.AttributeProto.Builder builder = Onnx.AttributeProto.newBuilder()
                .setName(name)
                .setType(Onnx.AttributeProto.AttributeType.FLOATS);
        for (float v : values) {
            builder.addFloats(v);
        }
        return builder.build();
    }

    public static Onnx.AttributeProto tensorAttr(String name, INDArray value) {
        return Onnx.AttributeProto.newBuilder()
                .setName(name)
                .setType(Onnx.AttributeProto.AttributeType.TENSOR)
                .setT(OnnxTensorUtils.toTensorProto(value))
                .build();
    }

    // --- Convenience methods for common op patterns ---

    /**
     * Build a unary float op test (single input, single output, same shape).
     */
    public static OpTestResult testUnaryOp(String opType, Path tempDir) throws Exception {
        return testUnaryOp(opType, tempDir, null);
    }

    /**
     * Build a unary float op test with attributes.
     */
    public static OpTestResult testUnaryOp(String opType, Path tempDir,
                                            Map<String, Onnx.AttributeProto> attributes) throws Exception {
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4).muli(2).subi(1); // range [-1, 1]
        return executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                attributes,
                tempDir,
                1e-4, 1e-6);
    }

    /**
     * Build a unary float op with a specific input range.
     */
    public static OpTestResult testUnaryOp(String opType, Path tempDir,
                                            Map<String, Onnx.AttributeProto> attributes,
                                            double inputMin, double inputMax) throws Exception {
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4)
                .muli(inputMax - inputMin).addi(inputMin);
        return executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                attributes,
                tempDir,
                1e-4, 1e-6);
    }

    /**
     * Build a binary float op test (two inputs, single output, broadcasting).
     */
    public static OpTestResult testBinaryOp(String opType, Path tempDir) throws Exception {
        return testBinaryOp(opType, tempDir, null);
    }

    /**
     * Build a binary float op test with attributes.
     */
    public static OpTestResult testBinaryOp(String opType, Path tempDir,
                                             Map<String, Onnx.AttributeProto> attributes) throws Exception {
        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3, 4).addi(0.5f); // avoid zeros for div
        INDArray b = Nd4j.rand(DataType.FLOAT, 2, 3, 4).addi(0.5f);
        return executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3, 4}, a),
                        InputSpec.graphInput("B", DataType.FLOAT, new long[]{2, 3, 4}, b)),
                List.of(new OutputSpec("C", DataType.FLOAT, new long[]{2, 3, 4})),
                attributes,
                tempDir,
                1e-4, 1e-6);
    }

    /**
     * Build a reduction op test.
     * In opset 13+, ReduceSum takes axes as input. Other reduce ops keep axes as attribute.
     */
    public static OpTestResult testReduceOp(String opType, Path tempDir,
                                             long[] axes, boolean keepdims) throws Exception {
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4).addi(0.1f);

        // Compute expected output shape
        long[] outputShape = computeReducedShape(new long[]{2, 3, 4}, axes, keepdims);

        Map<String, Onnx.AttributeProto> attrs = new LinkedHashMap<>();
        attrs.put("keepdims", intAttr("keepdims", keepdims ? 1 : 0));

        // ReduceSum (opset 13) takes axes as a second input, not an attribute
        boolean axesAsInput = "ReduceSum".equals(opType);

        List<InputSpec> inputs = new ArrayList<>();
        inputs.add(InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3, 4}, input));

        if (axesAsInput && axes != null) {
            INDArray axesArr = Nd4j.createFromArray(axes);
            inputs.add(InputSpec.constantInput("axes", DataType.LONG, new long[]{axes.length}, axesArr));
        } else if (!axesAsInput && axes != null) {
            attrs.put("axes", intsAttr("axes", axes));
        }

        return executeOpTest(
                opType,
                inputs,
                List.of(new OutputSpec("reduced", DataType.FLOAT, outputShape)),
                attrs,
                tempDir,
                1e-4, 1e-6);
    }

    /**
     * Compute shape after reduction.
     */
    private static long[] computeReducedShape(long[] inputShape, long[] axes, boolean keepdims) {
        if (axes == null) {
            return keepdims ? new long[inputShape.length] : new long[]{1};
        }
        Set<Integer> axisSet = new HashSet<>();
        for (long ax : axes) {
            int axis = (int) (ax < 0 ? inputShape.length + ax : ax);
            axisSet.add(axis);
        }

        List<Long> result = new ArrayList<>();
        for (int i = 0; i < inputShape.length; i++) {
            if (axisSet.contains(i)) {
                if (keepdims) {
                    result.add(1L);
                }
            } else {
                result.add(inputShape[i]);
            }
        }

        if (result.isEmpty()) {
            return new long[]{1};
        }
        return result.stream().mapToLong(Long::longValue).toArray();
    }

    // --- Data type conversion ---

    private static int dataTypeToOnnx(DataType dt) {
        switch (dt) {
            case FLOAT: return Onnx.TensorProto.DataType.FLOAT.getNumber();
            case DOUBLE: return Onnx.TensorProto.DataType.DOUBLE.getNumber();
            case INT: return Onnx.TensorProto.DataType.INT32.getNumber();
            case LONG: return Onnx.TensorProto.DataType.INT64.getNumber();
            case BOOL: return Onnx.TensorProto.DataType.BOOL.getNumber();
            case HALF: return Onnx.TensorProto.DataType.FLOAT16.getNumber();
            case BFLOAT16: return Onnx.TensorProto.DataType.BFLOAT16.getNumber();
            case SHORT: return Onnx.TensorProto.DataType.INT16.getNumber();
            case BYTE: return Onnx.TensorProto.DataType.INT8.getNumber();
            case UBYTE: return Onnx.TensorProto.DataType.UINT8.getNumber();
            case UINT16: return Onnx.TensorProto.DataType.UINT16.getNumber();
            case UINT32: return Onnx.TensorProto.DataType.UINT32.getNumber();
            case UINT64: return Onnx.TensorProto.DataType.UINT64.getNumber();
            default: throw new IllegalArgumentException("Unsupported DataType: " + dt);
        }
    }

    // --- Inner classes ---

    /**
     * Specification for a model input.
     */
    public static class InputSpec {
        public final String name;
        public final DataType dataType;
        public final long[] shape;
        public final INDArray data;         // runtime input data
        public final INDArray initializer;  // if non-null, this is a constant initializer

        private InputSpec(String name, DataType dataType, long[] shape, INDArray data, INDArray initializer) {
            this.name = name;
            this.dataType = dataType;
            this.shape = shape;
            this.data = data;
            this.initializer = initializer;
        }

        public static InputSpec graphInput(String name, DataType dataType, long[] shape, INDArray data) {
            return new InputSpec(name, dataType, shape, data, null);
        }

        public static InputSpec constantInput(String name, DataType dataType, long[] shape, INDArray initializer) {
            return new InputSpec(name, dataType, shape, null, initializer);
        }
    }

    /**
     * Specification for a model output.
     */
    public static class OutputSpec {
        public final String name;
        public final DataType dataType;
        public final long[] shape;

        public OutputSpec(String name, DataType dataType, long[] shape) {
            this.name = name;
            this.dataType = dataType;
            this.shape = shape;
        }
    }

    /**
     * Result of running an op test.
     */
    public static class OpTestResult {
        public final String opType;
        public File modelFile;
        public Map<String, INDArray> onnxRuntimeOutputs;
        public Map<String, INDArray> sameDiffOutputs;
        public Exception onnxRuntimeError;
        public Exception sameDiffError;
        public Throwable comparisonError;
        public boolean passed = false;

        public OpTestResult(String opType) {
            this.opType = opType;
        }

        public boolean onnxRuntimeSucceeded() {
            return onnxRuntimeError == null;
        }

        public boolean sameDiffSucceeded() {
            return sameDiffError == null;
        }

        public String getFailureReason() {
            if (onnxRuntimeError != null) return "ORT: " + onnxRuntimeError.getMessage();
            if (sameDiffError != null) return "SameDiff: " + sameDiffError.getMessage();
            if (comparisonError != null) return "Comparison: " + comparisonError.getMessage();
            return null;
        }
    }

}
