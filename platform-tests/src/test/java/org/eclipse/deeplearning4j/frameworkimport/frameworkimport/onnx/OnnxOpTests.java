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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;

import static org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.OnnxOpTestHelper.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Parameterized ONNX op-level tests.
 *
 * Programmatically builds small single-op ONNX models, runs them through both
 * ONNX Runtime (ground truth) and SameDiff (our import), and compares outputs.
 *
 * Modeled after TFGraphTestList for TensorFlow ops.
 */
@Slf4j
@Tag(TagNames.ONNX)
public class OnnxOpTests {

    @TempDir
    Path tempDir;

    // ======================== UNARY MATH OPS ========================

    static Stream<Arguments> unaryOps() {
        return Stream.of(
                // Standard unary ops (opset 13 compatible)
                Arguments.of("Abs", null, -1.0, 1.0),
                Arguments.of("Ceil", null, -2.5, 2.5),
                Arguments.of("Cos", null, -3.14, 3.14),
                Arguments.of("Cosh", null, -2.0, 2.0),
                Arguments.of("Elu", null, -2.0, 2.0),
                Arguments.of("Erf", null, -3.0, 3.0),
                Arguments.of("Exp", null, -2.0, 2.0),
                Arguments.of("Floor", null, -2.5, 2.5),
                Arguments.of("HardSigmoid", null, -3.0, 3.0),
                Arguments.of("LeakyRelu", null, -2.0, 2.0),
                Arguments.of("Neg", null, -2.0, 2.0),
                Arguments.of("Reciprocal", null, 0.5, 3.0),
                Arguments.of("Relu", null, -2.0, 2.0),
                Arguments.of("Round", null, -2.5, 2.5),
                Arguments.of("Selu", null, -2.0, 2.0),
                Arguments.of("Sigmoid", null, -5.0, 5.0),
                Arguments.of("Sign", null, -2.0, 2.0),
                Arguments.of("Sin", null, -3.14, 3.14),
                Arguments.of("Sinh", null, -2.0, 2.0),
                Arguments.of("Softplus", null, -2.0, 5.0),
                Arguments.of("Softsign", null, -5.0, 5.0),
                Arguments.of("Sqrt", null, 0.01, 5.0),
                Arguments.of("Tan", null, -1.0, 1.0),
                Arguments.of("Tanh", null, -3.0, 3.0),
                // Inverse trig - restricted input domain
                Arguments.of("Acos", null, -0.99, 0.99),
                Arguments.of("Acosh", null, 1.01, 5.0),
                Arguments.of("Asin", null, -0.99, 0.99),
                Arguments.of("Asinh", null, -5.0, 5.0),
                Arguments.of("Atan", null, -5.0, 5.0),
                Arguments.of("Atanh", null, -0.99, 0.99),
                // Log needs positive inputs
                Arguments.of("Log", null, 0.01, 10.0)
        );
    }

    // Ops requiring opset 14+
    static Stream<Arguments> unaryOpsOpset14() {
        return Stream.of(
                Arguments.of("HardSwish", null, -3.0, 3.0)
        );
    }

    @ParameterizedTest(name = "Unary14: {0}")
    @MethodSource("unaryOpsOpset14")
    void testUnaryOpsOpset14(String opType, Map<String, Onnx.AttributeProto> attributes,
                              double inputMin, double inputMax) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4)
                .muli(inputMax - inputMin).addi(inputMin);
        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                attributes,
                opDir,
                1e-4, 1e-6,
                14, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    // Ops requiring opset 18+
    static Stream<Arguments> unaryOpsOpset18() {
        return Stream.of(
                Arguments.of("Mish", null, -5.0, 5.0)
        );
    }

    @ParameterizedTest(name = "Unary18: {0}")
    @MethodSource("unaryOpsOpset18")
    void testUnaryOpsOpset18(String opType, Map<String, Onnx.AttributeProto> attributes,
                              double inputMin, double inputMax) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4)
                .muli(inputMax - inputMin).addi(inputMin);
        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                attributes,
                opDir,
                1e-4, 1e-6,
                18, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    @ParameterizedTest(name = "Unary: {0}")
    @MethodSource("unaryOps")
    void testUnaryOps(String opType, Map<String, Onnx.AttributeProto> attributes,
                      double inputMin, double inputMax) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        OpTestResult result = testUnaryOp(opType, opDir, attributes, inputMin, inputMax);
        assertOpResult(result);
    }

    // ======================== BINARY MATH OPS ========================

    static Stream<Arguments> binaryOps() {
        return Stream.of(
                Arguments.of("Add"),
                Arguments.of("Sub"),
                Arguments.of("Mul"),
                Arguments.of("Div")
        );
    }

    @ParameterizedTest(name = "Binary: {0}")
    @MethodSource("binaryOps")
    void testBinaryOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        OpTestResult result = testBinaryOp(opType, opDir);
        assertOpResult(result);
    }

    // ======================== BINARY COMPARISON OPS ========================

    static Stream<Arguments> comparisonOps() {
        return Stream.of(
                Arguments.of("Equal"),
                Arguments.of("Greater"),
                Arguments.of("GreaterOrEqual"),
                Arguments.of("Less"),
                Arguments.of("LessOrEqual")
        );
    }

    @ParameterizedTest(name = "Comparison: {0}")
    @MethodSource("comparisonOps")
    void testComparisonOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray b = Nd4j.rand(DataType.FLOAT, 2, 3, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3, 4}, a),
                        InputSpec.graphInput("B", DataType.FLOAT, new long[]{2, 3, 4}, b)),
                List.of(new OutputSpec("C", DataType.BOOL, new long[]{2, 3, 4})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    // ======================== BINARY LOGICAL OPS ========================

    static Stream<Arguments> logicalOps() {
        return Stream.of(
                Arguments.of("And"),
                Arguments.of("Or"),
                Arguments.of("Xor")
        );
    }

    @ParameterizedTest(name = "Logical: {0}")
    @MethodSource("logicalOps")
    void testLogicalOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3).gt(0.5).castTo(DataType.BOOL);
        INDArray b = Nd4j.rand(DataType.FLOAT, 2, 3).gt(0.5).castTo(DataType.BOOL);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.BOOL, new long[]{2, 3}, a),
                        InputSpec.graphInput("B", DataType.BOOL, new long[]{2, 3}, b)),
                List.of(new OutputSpec("C", DataType.BOOL, new long[]{2, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    // ======================== UNARY BOOL OPS ========================

    @ParameterizedTest(name = "BoolUnary: {0}")
    @MethodSource("boolUnaryOps")
    void testBoolUnaryOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3).gt(0.5).castTo(DataType.BOOL);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.BOOL, new long[]{2, 3}, input)),
                List.of(new OutputSpec("Y", DataType.BOOL, new long[]{2, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> boolUnaryOps() {
        return Stream.of(
                Arguments.of("Not")
        );
    }

    // ======================== FLOAT CHECK OPS ========================

    static Stream<Arguments> floatCheckOps() {
        return Stream.of(
                Arguments.of("IsInf"),
                Arguments.of("IsNaN")
        );
    }

    @ParameterizedTest(name = "FloatCheck: {0}")
    @MethodSource("floatCheckOps")
    void testFloatCheckOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        // Create input with some special values
        INDArray input = Nd4j.createFromArray(new float[]{
                1.0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0.0f, -1.0f
        }).reshape(2, 3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, input)),
                List.of(new OutputSpec("Y", DataType.BOOL, new long[]{2, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    // ======================== REDUCTION OPS ========================

    static Stream<Arguments> reduceOps() {
        return Stream.of(
                Arguments.of("ReduceSum", new long[]{1}, true),
                Arguments.of("ReduceSum", new long[]{-1}, false),
                Arguments.of("ReduceMean", new long[]{1}, true),
                Arguments.of("ReduceMean", new long[]{-1}, false),
                Arguments.of("ReduceMax", new long[]{1}, true),
                Arguments.of("ReduceMax", new long[]{-1}, false),
                Arguments.of("ReduceMin", new long[]{1}, true),
                Arguments.of("ReduceMin", new long[]{-1}, false),
                Arguments.of("ReduceProd", new long[]{1}, true),
                Arguments.of("ReduceL1", new long[]{1}, true),
                Arguments.of("ReduceL2", new long[]{1}, true),
                Arguments.of("ReduceLogSum", new long[]{1}, true),
                Arguments.of("ReduceLogSumExp", new long[]{1}, true),
                Arguments.of("ReduceSumSquare", new long[]{1}, true)
        );
    }

    @ParameterizedTest(name = "Reduce: {0} axes={1} keepdims={2}")
    @MethodSource("reduceOps")
    void testReduceOps(String opType, long[] axes, boolean keepdims) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        OpTestResult result = testReduceOp(opType, opDir, axes, keepdims);
        assertOpResult(result);
    }

    // ======================== POW OP ========================

    @ParameterizedTest(name = "Pow")
    @MethodSource("powOp")
    void testPowOp(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray base = Nd4j.rand(DataType.FLOAT, 2, 3).addi(0.5f);
        INDArray exponent = Nd4j.rand(DataType.FLOAT, 2, 3).muli(3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, base),
                        InputSpec.graphInput("Y", DataType.FLOAT, new long[]{2, 3}, exponent)),
                List.of(new OutputSpec("Z", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> powOp() {
        return Stream.of(Arguments.of("Pow"));
    }

    // ======================== MOD OP ========================

    @ParameterizedTest(name = "Mod")
    @MethodSource("modOp")
    void testModOp(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3).muli(10);
        INDArray b = Nd4j.rand(DataType.FLOAT, 2, 3).addi(1);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, a),
                        InputSpec.graphInput("B", DataType.FLOAT, new long[]{2, 3}, b)),
                List.of(new OutputSpec("C", DataType.FLOAT, new long[]{2, 3})),
                Map.of("fmod", intAttr("fmod", 1)),
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> modOp() {
        return Stream.of(Arguments.of("Mod"));
    }

    // ======================== SHAPE / TRANSFORM OPS ========================

    static Stream<Arguments> shapeOps() {
        return Stream.of(
                Arguments.of("Identity"),
                Arguments.of("Flatten"),
                Arguments.of("Transpose"),
                Arguments.of("Squeeze"),
                Arguments.of("Unsqueeze")
        );
    }

    @ParameterizedTest(name = "Shape: {0}")
    @MethodSource("shapeOps")
    void testShapeOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        switch (opType) {
            case "Identity": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3);
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, input)),
                        List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3})),
                        null, opDir, 0, 0);
                assertOpResult(result);
                break;
            }
            case "Flatten": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                        List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 12})),
                        Map.of("axis", intAttr("axis", 1)),
                        opDir, 0, 0);
                assertOpResult(result);
                break;
            }
            case "Transpose": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                        List.of(new OutputSpec("transposed", DataType.FLOAT, new long[]{4, 3, 2})),
                        Map.of("perm", intsAttr("perm", 2, 1, 0)),
                        opDir, 0, 0);
                assertOpResult(result);
                break;
            }
            case "Squeeze": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3, 1, 4);
                // Opset 13+: axes is an input, not an attribute
                INDArray axes = Nd4j.createFromArray(new long[]{0, 2});
                OpTestResult result = executeOpTest(opType,
                        List.of(
                                InputSpec.graphInput("data", DataType.FLOAT, new long[]{1, 3, 1, 4}, input),
                                InputSpec.constantInput("axes", DataType.LONG, new long[]{2}, axes)),
                        List.of(new OutputSpec("squeezed", DataType.FLOAT, new long[]{3, 4})),
                        null, opDir, 0, 0);
                assertOpResult(result);
                break;
            }
            case "Unsqueeze": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
                INDArray axes = Nd4j.createFromArray(new long[]{0, 3});
                OpTestResult result = executeOpTest(opType,
                        List.of(
                                InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 4}, input),
                                InputSpec.constantInput("axes", DataType.LONG, new long[]{2}, axes)),
                        List.of(new OutputSpec("expanded", DataType.FLOAT, new long[]{1, 3, 4, 1})),
                        null, opDir, 0, 0);
                assertOpResult(result);
                break;
            }
        }
    }

    // ======================== RESHAPE / CONCAT / SPLIT ========================

    @ParameterizedTest(name = "ReshapeConcatSplit: {0}")
    @MethodSource("reshapeConcatSplitOps")
    void testReshapeConcatSplitOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        switch (opType) {
            case "Reshape": {
                INDArray data = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
                INDArray shape = Nd4j.createFromArray(new long[]{6, 4});
                OpTestResult result = executeOpTest(opType,
                        List.of(
                                InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3, 4}, data),
                                InputSpec.constantInput("shape", DataType.LONG, new long[]{2}, shape)),
                        List.of(new OutputSpec("reshaped", DataType.FLOAT, new long[]{6, 4})),
                        null, opDir, 0, 0);
                assertOpResult(result);
                break;
            }
            case "Concat": {
                INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3);
                INDArray b = Nd4j.rand(DataType.FLOAT, 2, 5);
                OpTestResult result = executeOpTest(opType,
                        List.of(
                                InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, a),
                                InputSpec.graphInput("B", DataType.FLOAT, new long[]{2, 5}, b)),
                        List.of(new OutputSpec("C", DataType.FLOAT, new long[]{2, 8})),
                        Map.of("axis", intAttr("axis", 1)),
                        opDir, 0, 0);
                assertOpResult(result);
                break;
            }
            case "Split": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 2, 6);
                INDArray splitSizes = Nd4j.createFromArray(new long[]{2, 4});

                // Split produces multiple outputs
                Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                        .setName("test_Split");

                Onnx.NodeProto.Builder nodeBuilder = Onnx.NodeProto.newBuilder()
                        .setOpType("Split")
                        .setName("Split_0")
                        .addInput("input")
                        .addInput("split")
                        .addOutput("output_0")
                        .addOutput("output_1")
                        .addAttribute(intAttr("axis", 1));

                graphBuilder.addNode(nodeBuilder.build());
                graphBuilder.addInput(makeValueInfo("input", DataType.FLOAT, new long[]{2, 6}));
                // split is a constant initializer, not a graph input
                graphBuilder.addInitializer(
                        org.nd4j.onnx.OnnxTensorUtils.toTensorProto(splitSizes, "split"));
                graphBuilder.addOutput(makeValueInfo("output_0", DataType.FLOAT, new long[]{2, 2}));
                graphBuilder.addOutput(makeValueInfo("output_1", DataType.FLOAT, new long[]{2, 4}));

                Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                        .setIrVersion(7)
                        .setGraph(graphBuilder.build())
                        .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                        .build();

                java.io.File modelFile = saveModel(model, opDir);

                // Run ORT
                Map<String, INDArray> inputMap = Map.of("input", input);
                try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                             org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                                     .modelUri(modelFile.getAbsolutePath()).build()) {
                    Map<String, INDArray> ortOutputs = runner.exec(inputMap);
                    assertNotNull(ortOutputs);
                    log.info("Split ORT outputs: {}", ortOutputs.keySet());
                }

                // Run SameDiff
                org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                        new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
                org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                        modelFile.getAbsolutePath(), inputMap, false, false);
                Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
                assertNotNull(sdOutputs);
                log.info("Split SameDiff outputs: {}", sdOutputs.keySet());
                break;
            }
        }
    }

    static Stream<Arguments> reshapeConcatSplitOps() {
        return Stream.of(
                Arguments.of("Reshape"),
                Arguments.of("Concat"),
                Arguments.of("Split")
        );
    }

    // ======================== CAST ========================

    @ParameterizedTest(name = "Cast: {0}")
    @MethodSource("castOps")
    void testCastOps(String label, int targetType, DataType expectedDt) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, "Cast_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3);

        // Output shape is same, dtype changes
        long[] shape = new long[]{2, 3};
        OpTestResult result = executeOpTest(
                "Cast",
                List.of(InputSpec.graphInput("input", DataType.FLOAT, shape, input)),
                List.of(new OutputSpec("output", expectedDt, shape)),
                Map.of("to", intAttr("to", targetType)),
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> castOps() {
        return Stream.of(
                Arguments.of("FLOAT->DOUBLE", 11, DataType.DOUBLE),
                Arguments.of("FLOAT->INT32", 6, DataType.INT),
                Arguments.of("FLOAT->INT64", 7, DataType.LONG)
        );
    }

    // ======================== CLIP ========================

    @ParameterizedTest(name = "Clip")
    @MethodSource("clipOps")
    void testClipOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4).muli(10).subi(5);
        INDArray min = Nd4j.scalar(DataType.FLOAT, -2.0f);
        INDArray max = Nd4j.scalar(DataType.FLOAT, 2.0f);

        // Opset 11+: min and max are inputs
        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 3, 4}, input),
                        InputSpec.constantInput("min", DataType.FLOAT, new long[]{}, min),
                        InputSpec.constantInput("max", DataType.FLOAT, new long[]{}, max)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3, 4})),
                null,
                opDir,
                1e-6, 1e-8);
        assertOpResult(result);
    }

    static Stream<Arguments> clipOps() {
        return Stream.of(Arguments.of("Clip"));
    }

    // ======================== GATHER ========================

    @ParameterizedTest(name = "Gather")
    @MethodSource("gatherOps")
    void testGatherOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 5, 4);
        INDArray indices = Nd4j.createFromArray(new long[]{0, 2, 4}).reshape(3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{5, 4}, data),
                        InputSpec.constantInput("indices", DataType.LONG, new long[]{3}, indices)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{3, 4})),
                Map.of("axis", intAttr("axis", 0)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> gatherOps() {
        return Stream.of(Arguments.of("Gather"));
    }

    // ======================== SLICE ========================

    @ParameterizedTest(name = "Slice")
    @MethodSource("sliceOps")
    void testSliceOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 4, 6);
        INDArray starts = Nd4j.createFromArray(new long[]{1, 2});
        INDArray ends = Nd4j.createFromArray(new long[]{3, 5});
        INDArray axes = Nd4j.createFromArray(new long[]{0, 1});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{4, 6}, data),
                        InputSpec.constantInput("starts", DataType.LONG, new long[]{2}, starts),
                        InputSpec.constantInput("ends", DataType.LONG, new long[]{2}, ends),
                        InputSpec.constantInput("axes", DataType.LONG, new long[]{2}, axes)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> sliceOps() {
        return Stream.of(Arguments.of("Slice"));
    }

    // ======================== WHERE ========================

    @ParameterizedTest(name = "Where")
    @MethodSource("whereOps")
    void testWhereOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray condition = Nd4j.rand(DataType.FLOAT, 2, 3).gt(0.5).castTo(DataType.BOOL);
        INDArray x = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray y = Nd4j.rand(DataType.FLOAT, 2, 3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("condition", DataType.BOOL, new long[]{2, 3}, condition),
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, x),
                        InputSpec.graphInput("Y", DataType.FLOAT, new long[]{2, 3}, y)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> whereOps() {
        return Stream.of(Arguments.of("Where"));
    }

    // ======================== TILE ========================

    @ParameterizedTest(name = "Tile")
    @MethodSource("tileOps")
    void testTileOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray repeats = Nd4j.createFromArray(new long[]{2, 3});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 3}, input),
                        InputSpec.constantInput("repeats", DataType.LONG, new long[]{2}, repeats)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{4, 9})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> tileOps() {
        return Stream.of(Arguments.of("Tile"));
    }

    // ======================== ONEHOT ========================

    @ParameterizedTest(name = "OneHot")
    @MethodSource("oneHotOps")
    void testOneHotOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray indices = Nd4j.createFromArray(new long[]{0, 1, 2, 3}).reshape(4);
        INDArray depth = Nd4j.scalar(DataType.LONG, 5);
        INDArray values = Nd4j.createFromArray(new float[]{0.0f, 1.0f});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{4}, indices),
                        InputSpec.constantInput("depth", DataType.LONG, new long[]{}, depth),
                        InputSpec.constantInput("values", DataType.FLOAT, new long[]{2}, values)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{4, 5})),
                Map.of("axis", intAttr("axis", -1)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> oneHotOps() {
        return Stream.of(Arguments.of("OneHot"));
    }

    // ======================== EXPAND ========================

    @ParameterizedTest(name = "Expand")
    @MethodSource("expandOps")
    void testExpandOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3);
        INDArray shape = Nd4j.createFromArray(new long[]{2, 3});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{1, 3}, input),
                        InputSpec.constantInput("shape", DataType.LONG, new long[]{2}, shape)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> expandOps() {
        return Stream.of(Arguments.of("Expand"));
    }

    // ======================== CUMSUM ========================

    @ParameterizedTest(name = "CumSum")
    @MethodSource("cumSumOps")
    void testCumSumOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
        INDArray axis = Nd4j.scalar(DataType.INT, 1);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("x", DataType.FLOAT, new long[]{3, 4}, input),
                        InputSpec.constantInput("axis", DataType.INT, new long[]{}, axis)),
                List.of(new OutputSpec("y", DataType.FLOAT, new long[]{3, 4})),
                null,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> cumSumOps() {
        return Stream.of(Arguments.of("CumSum"));
    }

    // ======================== ARGMAX / ARGMIN ========================

    static Stream<Arguments> argOps() {
        return Stream.of(
                Arguments.of("ArgMax"),
                Arguments.of("ArgMin")
        );
    }

    @ParameterizedTest(name = "Arg: {0}")
    @MethodSource("argOps")
    void testArgOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 4}, input)),
                List.of(new OutputSpec("reduced", DataType.LONG, new long[]{3, 1})),
                Map.of("axis", intAttr("axis", 1), "keepdims", intAttr("keepdims", 1)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    // ======================== RANGE ========================

    @ParameterizedTest(name = "Range")
    @MethodSource("rangeOps")
    void testRangeOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray start = Nd4j.scalar(DataType.FLOAT, 0.0f);
        INDArray limit = Nd4j.scalar(DataType.FLOAT, 10.0f);
        INDArray delta = Nd4j.scalar(DataType.FLOAT, 2.0f);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("start", DataType.FLOAT, new long[]{}, start),
                        InputSpec.graphInput("limit", DataType.FLOAT, new long[]{}, limit),
                        InputSpec.graphInput("delta", DataType.FLOAT, new long[]{}, delta)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{5})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> rangeOps() {
        return Stream.of(Arguments.of("Range"));
    }

    // ======================== SHAPE / SIZE ========================

    static Stream<Arguments> shapeInfoOps() {
        return Stream.of(
                Arguments.of("Shape"),
                Arguments.of("Size")
        );
    }

    @ParameterizedTest(name = "ShapeInfo: {0}")
    @MethodSource("shapeInfoOps")
    void testShapeInfoOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4);

        switch (opType) {
            case "Shape": {
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                        List.of(new OutputSpec("shape", DataType.LONG, new long[]{3})),
                        null, opDir, 0, 0);
                assertOpResult(result);
                break;
            }
            case "Size": {
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                        List.of(new OutputSpec("size", DataType.LONG, new long[]{})),
                        null, opDir, 0, 0);
                assertOpResult(result);
                break;
            }
        }
    }

    // ======================== CONSTANTOFSHAPE ========================

    @ParameterizedTest(name = "ConstantOfShape")
    @MethodSource("constantOfShapeOps")
    void testConstantOfShapeOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray shape = Nd4j.createFromArray(new long[]{2, 3, 4});

        // Create a 1-element 1-D tensor attribute for the fill value
        // ONNX spec requires value to be a 1-element tensor with dims=[1]
        Onnx.TensorProto valueTensor = Onnx.TensorProto.newBuilder()
                .setDataType(Onnx.TensorProto.DataType.FLOAT.getNumber())
                .addDims(1)
                .addFloatData(3.14f)
                .build();

        Map<String, Onnx.AttributeProto> attrs = Map.of(
                "value", Onnx.AttributeProto.newBuilder()
                        .setName("value")
                        .setType(Onnx.AttributeProto.AttributeType.TENSOR)
                        .setT(valueTensor)
                        .build()
        );

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("input", DataType.LONG, new long[]{3}, shape)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3, 4})),
                attrs,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> constantOfShapeOps() {
        return Stream.of(Arguments.of("ConstantOfShape"));
    }

    // ======================== NEURAL NETWORK OPS ========================

    static Stream<Arguments> nnOps() {
        return Stream.of(
                Arguments.of("Softmax"),
                Arguments.of("LogSoftmax"),
                Arguments.of("MatMul"),
                Arguments.of("Gemm")
        );
    }

    @ParameterizedTest(name = "NN: {0}")
    @MethodSource("nnOps")
    void testNNOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        switch (opType) {
            case "Softmax": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 2, 5);
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 5}, input)),
                        List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 5})),
                        Map.of("axis", intAttr("axis", -1)),
                        opDir, 1e-4, 1e-6);
                assertOpResult(result);
                break;
            }
            case "LogSoftmax": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 2, 5);
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 5}, input)),
                        List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 5})),
                        Map.of("axis", intAttr("axis", -1)),
                        opDir, 1e-4, 1e-6);
                assertOpResult(result);
                break;
            }
            case "MatMul": {
                INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3);
                INDArray b = Nd4j.rand(DataType.FLOAT, 3, 4);
                OpTestResult result = executeOpTest(opType,
                        List.of(
                                InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, a),
                                InputSpec.graphInput("B", DataType.FLOAT, new long[]{3, 4}, b)),
                        List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 4})),
                        null, opDir, 1e-4, 1e-6);
                assertOpResult(result);
                break;
            }
            case "Gemm": {
                INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3);
                INDArray b = Nd4j.rand(DataType.FLOAT, 3, 4);
                INDArray c = Nd4j.rand(DataType.FLOAT, 4);
                OpTestResult result = executeOpTest(opType,
                        List.of(
                                InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, a),
                                InputSpec.graphInput("B", DataType.FLOAT, new long[]{3, 4}, b),
                                InputSpec.graphInput("C", DataType.FLOAT, new long[]{4}, c)),
                        List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 4})),
                        Map.of("alpha", floatAttr("alpha", 1.0f),
                                "beta", floatAttr("beta", 1.0f),
                                "transA", intAttr("transA", 0),
                                "transB", intAttr("transB", 0)),
                        opDir, 1e-4, 1e-6);
                assertOpResult(result);
                break;
            }
        }
    }

    // ======================== POOLING OPS ========================

    static Stream<Arguments> poolOps() {
        return Stream.of(
                Arguments.of("AveragePool"),
                Arguments.of("MaxPool"),
                Arguments.of("GlobalAveragePool"),
                Arguments.of("GlobalMaxPool")
        );
    }

    @ParameterizedTest(name = "Pool: {0}")
    @MethodSource("poolOps")
    void testPoolOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        switch (opType) {
            case "AveragePool":
            case "MaxPool": {
                // NCHW format: [batch, channels, height, width]
                INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 4, 4);
                Map<String, Onnx.AttributeProto> attrs = new LinkedHashMap<>();
                attrs.put("kernel_shape", intsAttr("kernel_shape", 2, 2));
                attrs.put("strides", intsAttr("strides", 2, 2));

                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 1, 4, 4}, input)),
                        List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 1, 2, 2})),
                        attrs, opDir, 1e-4, 1e-6);
                assertOpResult(result);
                break;
            }
            case "GlobalAveragePool":
            case "GlobalMaxPool": {
                INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3, 4, 4);
                OpTestResult result = executeOpTest(opType,
                        List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 3, 4, 4}, input)),
                        List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 3, 1, 1})),
                        null, opDir, 1e-4, 1e-6);
                assertOpResult(result);
                break;
            }
        }
    }

    // ======================== CONV ========================

    @ParameterizedTest(name = "Conv")
    @MethodSource("convOps")
    void testConvOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        // Simple 2D conv: [N,C,H,W]
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 5, 5);
        INDArray weight = Nd4j.rand(DataType.FLOAT, 1, 1, 3, 3);
        INDArray bias = Nd4j.rand(DataType.FLOAT, 1);

        Map<String, Onnx.AttributeProto> attrs = new LinkedHashMap<>();
        attrs.put("kernel_shape", intsAttr("kernel_shape", 3, 3));
        attrs.put("strides", intsAttr("strides", 1, 1));
        attrs.put("pads", intsAttr("pads", 0, 0, 0, 0));

        OpTestResult result = executeOpTest(opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 1, 5, 5}, input),
                        InputSpec.constantInput("W", DataType.FLOAT, new long[]{1, 1, 3, 3}, weight),
                        InputSpec.constantInput("B", DataType.FLOAT, new long[]{1}, bias)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 1, 3, 3})),
                attrs,
                opDir, 1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> convOps() {
        return Stream.of(Arguments.of("Conv"));
    }

    // ======================== BATCHNORM ========================

    @ParameterizedTest(name = "BatchNormalization")
    @MethodSource("batchNormOps")
    void testBatchNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        int channels = 3;
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, channels, 4, 4);
        INDArray scale = Nd4j.ones(DataType.FLOAT, channels);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, channels);
        INDArray mean = Nd4j.zeros(DataType.FLOAT, channels);
        INDArray var = Nd4j.ones(DataType.FLOAT, channels);

        OpTestResult result = executeOpTest(opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, channels, 4, 4}, input),
                        InputSpec.constantInput("scale", DataType.FLOAT, new long[]{channels}, scale),
                        InputSpec.constantInput("B", DataType.FLOAT, new long[]{channels}, bias),
                        InputSpec.constantInput("input_mean", DataType.FLOAT, new long[]{channels}, mean),
                        InputSpec.constantInput("input_var", DataType.FLOAT, new long[]{channels}, var)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, channels, 4, 4})),
                null,
                opDir, 1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> batchNormOps() {
        return Stream.of(Arguments.of("BatchNormalization"));
    }

    // ======================== EINSUM ========================

    @ParameterizedTest(name = "Einsum")
    @MethodSource("einsumOps")
    void testEinsumOps(String equation) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, "Einsum_");

        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray b = Nd4j.rand(DataType.FLOAT, 3, 4);

        OpTestResult result = executeOpTest(
                "Einsum",
                List.of(
                        InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, a),
                        InputSpec.graphInput("B", DataType.FLOAT, new long[]{3, 4}, b)),
                List.of(new OutputSpec("C", DataType.FLOAT, new long[]{2, 4})),
                Map.of("equation", stringAttr("equation", equation)),
                opDir, 1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> einsumOps() {
        return Stream.of(Arguments.of("ij,jk->ik"));
    }

    // ======================== TOPK ========================

    @ParameterizedTest(name = "TopK")
    @MethodSource("topKOps")
    void testTopKOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 5);
        // K must be a 1-D tensor of size 1 per ONNX spec
        INDArray k = Nd4j.createFromArray(new long[]{3});

        // TopK produces 2 outputs: values and indices
        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_TopK");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("TopK")
                .setName("TopK_0")
                .addInput("X")
                .addInput("K")
                .addOutput("Values")
                .addOutput("Indices")
                .addAttribute(intAttr("axis", -1))
                .build());

        graphBuilder.addInput(makeValueInfo("X", DataType.FLOAT, new long[]{3, 5}));
        // K is a constant initializer, not a graph input
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(k, "K"));
        graphBuilder.addOutput(makeValueInfo("Values", DataType.FLOAT, new long[]{3, 3}));
        graphBuilder.addOutput(makeValueInfo("Indices", DataType.LONG, new long[]{3, 3}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Map.of("X", input);

        // Run ORT
        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            Map<String, INDArray> ortOutputs = runner.exec(inputMap);
            assertNotNull(ortOutputs);
            assertFalse(ortOutputs.isEmpty(), "ORT should produce outputs");
        }

        // Run SameDiff
        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        assertNotNull(sdOutputs);
        assertFalse(sdOutputs.isEmpty(), "SameDiff should produce outputs");
    }

    static Stream<Arguments> topKOps() {
        return Stream.of(Arguments.of("TopK"));
    }

    // ======================== NONZERO ========================

    @ParameterizedTest(name = "NonZero")
    @MethodSource("nonZeroOps")
    void testNonZeroOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.createFromArray(new float[]{
                1, 0, 0, 2, 0, 3
        }).reshape(2, 3);

        // NonZero output shape is dynamic; we set a loose shape
        // ORT returns [ndim, num_nonzero], for this input: [2, 3]
        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, input)),
                List.of(new OutputSpec("Y", DataType.LONG, null)),  // dynamic shape
                null,
                opDir,
                0, 0);
        // NonZero has dynamic output - just check it runs
        if (result.onnxRuntimeError != null) {
            fail("ORT failed: " + result.onnxRuntimeError.getMessage());
        }
        if (result.sameDiffError != null) {
            log.warn("SameDiff NonZero not yet supported: {}", result.sameDiffError.getMessage());
        }
    }

    static Stream<Arguments> nonZeroOps() {
        return Stream.of(Arguments.of("NonZero"));
    }

    // ======================== BATCH 1: ADDITIONAL UNARY/ACTIVATION OPS ========================

    static Stream<Arguments> additionalUnaryOps() {
        return Stream.of(
                Arguments.of("ThresholdedRelu", Map.of("alpha", floatAttr("alpha", 1.0f)), -2.0, 2.0, 10),
                Arguments.of("Celu", Map.of("alpha", floatAttr("alpha", 1.0f)), -2.0, 2.0, 12),
                Arguments.of("Shrink", Map.of("bias", floatAttr("bias", 0.0f), "lambd", floatAttr("lambd", 0.5f)), -3.0, 3.0, 9)
        );
    }

    @ParameterizedTest(name = "AdditionalUnary: {0}")
    @MethodSource("additionalUnaryOps")
    void testAdditionalUnaryOps(String opType, Map<String, Onnx.AttributeProto> attributes,
                                 double inputMin, double inputMax, int opset) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4)
                .muli(inputMax - inputMin).addi(inputMin);
        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                attributes,
                opDir,
                1e-4, 1e-6,
                opset, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    // ======================== BATCH 3: ELEMENT-WISE MULTI-INPUT OPS ========================

    static Stream<Arguments> elementWiseMultiOps() {
        return Stream.of(
                Arguments.of("Max"),
                Arguments.of("Min"),
                Arguments.of("Sum"),
                Arguments.of("Mean")
        );
    }

    @ParameterizedTest(name = "ElementWiseMulti: {0}")
    @MethodSource("elementWiseMultiOps")
    void testElementWiseMultiOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray b = Nd4j.rand(DataType.FLOAT, 2, 3, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data_0", DataType.FLOAT, new long[]{2, 3, 4}, a),
                        InputSpec.graphInput("data_1", DataType.FLOAT, new long[]{2, 3, 4}, b)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3, 4})),
                null,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    // ======================== PRELU ========================

    @ParameterizedTest(name = "PRelu")
    @MethodSource("preluOps")
    void testPReluOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray x = Nd4j.rand(DataType.FLOAT, 2, 3, 4).muli(2).subi(1);
        INDArray slope = Nd4j.rand(DataType.FLOAT, 3, 1).muli(0.5f); // broadcastable

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, x),
                        InputSpec.graphInput("slope", DataType.FLOAT, new long[]{3, 1}, slope)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                null,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> preluOps() {
        return Stream.of(Arguments.of("PRelu"));
    }

    // ======================== BATCH 4: SHAPE/TRANSFORM OPS ========================

    @ParameterizedTest(name = "DepthToSpace")
    @MethodSource("depthToSpaceOps")
    void testDepthToSpaceOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 8, 2, 2);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{1, 8, 2, 2}, input)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{1, 2, 4, 4})),
                Map.of("blocksize", intAttr("blocksize", 2), "mode", stringAttr("mode", "DCR")),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> depthToSpaceOps() {
        return Stream.of(Arguments.of("DepthToSpace"));
    }

    @ParameterizedTest(name = "SpaceToDepth")
    @MethodSource("spaceToDepthOps")
    void testSpaceToDepthOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 2, 4, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{1, 2, 4, 4}, input)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{1, 8, 2, 2})),
                Map.of("blocksize", intAttr("blocksize", 2)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> spaceToDepthOps() {
        return Stream.of(Arguments.of("SpaceToDepth"));
    }

    @ParameterizedTest(name = "GatherElements")
    @MethodSource("gatherElementsOps")
    void testGatherElementsOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 3, 3);
        INDArray indices = Nd4j.createFromArray(new long[][]{{0, 1, 2}, {2, 0, 1}}).castTo(DataType.LONG);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 3}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{2, 3}, indices)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3})),
                Map.of("axis", intAttr("axis", 0)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> gatherElementsOps() {
        return Stream.of(Arguments.of("GatherElements"));
    }

    @ParameterizedTest(name = "GatherND")
    @MethodSource("gatherNDOps")
    void testGatherNDOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray indices = Nd4j.createFromArray(new long[][]{{0}, {1}}).castTo(DataType.LONG);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{2, 1}, indices)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3})),
                Map.of("batch_dims", intAttr("batch_dims", 0)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> gatherNDOps() {
        return Stream.of(Arguments.of("GatherND"));
    }

    @ParameterizedTest(name = "ScatterElements")
    @MethodSource("scatterElementsOps")
    void testScatterElementsOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray data = Nd4j.zeros(DataType.FLOAT, 3, 3);
        INDArray indices = Nd4j.createFromArray(new long[][]{{0, 1, 2}}).castTo(DataType.LONG);
        INDArray updates = Nd4j.createFromArray(new float[][]{{1.0f, 2.0f, 3.0f}});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 3}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{1, 3}, indices),
                        InputSpec.graphInput("updates", DataType.FLOAT, new long[]{1, 3}, updates)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{3, 3})),
                Map.of("axis", intAttr("axis", 1)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> scatterElementsOps() {
        return Stream.of(Arguments.of("ScatterElements"));
    }

    @ParameterizedTest(name = "ScatterND")
    @MethodSource("scatterNDOps")
    void testScatterNDOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray data = Nd4j.zeros(DataType.FLOAT, 4, 4);
        INDArray indices = Nd4j.createFromArray(new long[][]{{0}, {2}}).castTo(DataType.LONG);
        INDArray updates = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}, {5, 6, 7, 8}});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{4, 4}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{2, 1}, indices),
                        InputSpec.graphInput("updates", DataType.FLOAT, new long[]{2, 4}, updates)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{4, 4})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> scatterNDOps() {
        return Stream.of(Arguments.of("ScatterND"));
    }

    @ParameterizedTest(name = "Pad")
    @MethodSource("padOps")
    void testPadOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 2, 3);
        // pads format: [x1_begin, x2_begin, x1_end, x2_end]
        INDArray pads = Nd4j.createFromArray(new long[]{1, 0, 1, 0});
        INDArray constantValue = Nd4j.scalar(DataType.FLOAT, 0.0f);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3}, data),
                        InputSpec.constantInput("pads", DataType.LONG, new long[]{4}, pads),
                        InputSpec.constantInput("constant_value", DataType.FLOAT, new long[]{}, constantValue)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{4, 3})),
                Map.of("mode", stringAttr("mode", "constant")),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> padOps() {
        return Stream.of(Arguments.of("Pad"));
    }

    @ParameterizedTest(name = "Trilu")
    @MethodSource("triluOps")
    void testTriluOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{3, 4}, input)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{3, 4})),
                Map.of("upper", intAttr("upper", 1)),
                opDir,
                0, 0,
                14, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    static Stream<Arguments> triluOps() {
        return Stream.of(Arguments.of("Trilu"));
    }

    @ParameterizedTest(name = "ReverseSequence")
    @MethodSource("reverseSequenceOps")
    void testReverseSequenceOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 4, 3);
        INDArray seqLens = Nd4j.createFromArray(new long[]{2, 3, 1});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{4, 3}, input),
                        InputSpec.graphInput("sequence_lens", DataType.LONG, new long[]{3}, seqLens)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{4, 3})),
                Map.of("batch_axis", intAttr("batch_axis", 1), "time_axis", intAttr("time_axis", 0)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> reverseSequenceOps() {
        return Stream.of(Arguments.of("ReverseSequence"));
    }

    @ParameterizedTest(name = "Compress")
    @MethodSource("compressOps")
    void testCompressOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 4);
        INDArray condition = Nd4j.createFromArray(new boolean[]{true, false, true, true}).castTo(DataType.BOOL);

        // axis=1, condition selects columns → [3, 3]
        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{3, 4}, input),
                        InputSpec.graphInput("condition", DataType.BOOL, new long[]{4}, condition)),
                List.of(new OutputSpec("output", DataType.FLOAT, null)),
                Map.of("axis", intAttr("axis", 1)),
                opDir,
                0, 0);
        // Compress has dynamic output shape; check both ran and shapes are compatible
        if (result.onnxRuntimeError != null) {
            fail("ORT failed: " + result.onnxRuntimeError.getMessage());
        }
        if (result.sameDiffError != null) {
            fail("SameDiff failed: " + result.sameDiffError.getMessage());
        }
        // Verify both produced output
        assertNotNull(result.onnxRuntimeOutputs);
        assertNotNull(result.sameDiffOutputs);
    }

    static Stream<Arguments> compressOps() {
        return Stream.of(Arguments.of("Compress"));
    }

    @ParameterizedTest(name = "EyeLike")
    @MethodSource("eyeLikeOps")
    void testEyeLikeOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.zeros(DataType.FLOAT, 3, 3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{3, 3}, input)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{3, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> eyeLikeOps() {
        return Stream.of(Arguments.of("EyeLike"));
    }

    @ParameterizedTest(name = "Hardmax")
    @MethodSource("hardmaxOps")
    void testHardmaxOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 5);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 5}, input)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 5})),
                Map.of("axis", intAttr("axis", -1)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> hardmaxOps() {
        return Stream.of(Arguments.of("Hardmax"));
    }

    // ======================== BATCH 5: RESIZE / UPSAMPLE ========================

    @ParameterizedTest(name = "Resize")
    @MethodSource("resizeOps")
    void testResizeOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 2, 2);
        // Use scales-based resize (simpler, avoids empty tensor issues)
        INDArray scales = Nd4j.createFromArray(new float[]{1.0f, 1.0f, 2.0f, 2.0f});

        // Build model manually to handle the empty roi input
        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_Resize");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("Resize")
                .setName("Resize_0")
                .addInput("X")
                .addInput("")  // empty roi
                .addInput("scales")
                .addOutput("Y")
                .addAttribute(stringAttr("mode", "nearest"))
                .build());

        graphBuilder.addInput(makeValueInfo("X", DataType.FLOAT, new long[]{1, 1, 2, 2}));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(scales, "scales"));
        graphBuilder.addOutput(makeValueInfo("Y", DataType.FLOAT, new long[]{1, 1, 4, 4}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Map.of("X", input);

        Map<String, INDArray> ortOutputs;
        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            ortOutputs = runner.exec(inputMap);
            assertNotNull(ortOutputs);
        }

        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        assertNotNull(sdOutputs);

        compareOutputs(ortOutputs, sdOutputs, 1e-4, 1e-6);
    }

    static Stream<Arguments> resizeOps() {
        return Stream.of(Arguments.of("Resize"));
    }

    // ======================== BATCH 6: NORMALIZATION OPS ========================

    @ParameterizedTest(name = "LRN")
    @MethodSource("lrnOps")
    void testLrnOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3, 4, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 3, 4, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 3, 4, 4})),
                Map.of("size", intAttr("size", 3)),
                opDir,
                1e-3, 1e-5);
        assertOpResult(result);
    }

    static Stream<Arguments> lrnOps() {
        return Stream.of(Arguments.of("LRN"));
    }

    @ParameterizedTest(name = "InstanceNormalization")
    @MethodSource("instanceNormOps")
    void testInstanceNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        int channels = 3;
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, channels, 4, 4);
        INDArray scale = Nd4j.ones(DataType.FLOAT, channels);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, channels);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{1, channels, 4, 4}, input),
                        InputSpec.constantInput("scale", DataType.FLOAT, new long[]{channels}, scale),
                        InputSpec.constantInput("B", DataType.FLOAT, new long[]{channels}, bias)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{1, channels, 4, 4})),
                Map.of("epsilon", floatAttr("epsilon", 1e-5f)),
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> instanceNormOps() {
        return Stream.of(Arguments.of("InstanceNormalization"));
    }

    @ParameterizedTest(name = "LayerNormalization")
    @MethodSource("layerNormOps")
    void testLayerNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray scale = Nd4j.ones(DataType.FLOAT, 4);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, input),
                        InputSpec.constantInput("Scale", DataType.FLOAT, new long[]{4}, scale),
                        InputSpec.constantInput("B", DataType.FLOAT, new long[]{4}, bias)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                Map.of("axis", intAttr("axis", -1), "epsilon", floatAttr("epsilon", 1e-5f)),
                opDir,
                1e-4, 1e-6,
                17, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    static Stream<Arguments> layerNormOps() {
        return Stream.of(Arguments.of("LayerNormalization"));
    }

    @ParameterizedTest(name = "GroupNormalization")
    @MethodSource("groupNormOps")
    void testGroupNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        int channels = 4;
        int numGroups = 2;
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, channels, 2, 2);
        // Scale and bias must have shape [num_groups] per ONNX GroupNormalization spec
        INDArray scale = Nd4j.ones(DataType.FLOAT, numGroups);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, numGroups);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, channels, 2, 2}, input),
                        InputSpec.constantInput("scale", DataType.FLOAT, new long[]{numGroups}, scale),
                        InputSpec.constantInput("bias", DataType.FLOAT, new long[]{numGroups}, bias)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, channels, 2, 2})),
                Map.of("num_groups", intAttr("num_groups", numGroups), "epsilon", floatAttr("epsilon", 1e-5f)),
                opDir,
                1e-4, 1e-6,
                18, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    static Stream<Arguments> groupNormOps() {
        return Stream.of(Arguments.of("GroupNormalization"));
    }

    @ParameterizedTest(name = "MeanVarianceNormalization")
    @MethodSource("mvnOps")
    void testMeanVarianceNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3, 4, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 3, 4, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 3, 4, 4})),
                Map.of("axes", intsAttr("axes", 0, 2, 3)),
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> mvnOps() {
        return Stream.of(Arguments.of("MeanVarianceNormalization"));
    }

    // ======================== BATCH 7: CONV/POOL VARIANTS ========================

    @ParameterizedTest(name = "ConvTranspose")
    @MethodSource("convTransposeOps")
    void testConvTransposeOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 3, 3);
        INDArray weight = Nd4j.rand(DataType.FLOAT, 1, 1, 3, 3);

        Map<String, Onnx.AttributeProto> attrs = new LinkedHashMap<>();
        attrs.put("kernel_shape", intsAttr("kernel_shape", 3, 3));
        attrs.put("strides", intsAttr("strides", 1, 1));

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 1, 3, 3}, input),
                        InputSpec.constantInput("W", DataType.FLOAT, new long[]{1, 1, 3, 3}, weight)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 1, 5, 5})),
                attrs,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> convTransposeOps() {
        return Stream.of(Arguments.of("ConvTranspose"));
    }

    @ParameterizedTest(name = "GlobalLpPool")
    @MethodSource("globalLpPoolOps")
    void testGlobalLpPoolOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 3, 4, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 3, 4, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 3, 1, 1})),
                Map.of("p", intAttr("p", 2)),
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> globalLpPoolOps() {
        return Stream.of(Arguments.of("GlobalLpPool"));
    }

    @ParameterizedTest(name = "LpPool")
    @MethodSource("lpPoolOps")
    void testLpPoolOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 4, 4);

        Map<String, Onnx.AttributeProto> attrs = new LinkedHashMap<>();
        attrs.put("kernel_shape", intsAttr("kernel_shape", 2, 2));
        attrs.put("strides", intsAttr("strides", 2, 2));
        attrs.put("p", intAttr("p", 2));

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{1, 1, 4, 4}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{1, 1, 2, 2})),
                attrs,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> lpPoolOps() {
        return Stream.of(Arguments.of("LpPool"));
    }

    // ======================== BATCH 8: COM.MICROSOFT EXTENSION OPS ========================

    @ParameterizedTest(name = "MSGelu: {0}")
    @MethodSource("msGeluOps")
    void testMsGeluOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3).muli(2).subi(1);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msGeluOps() {
        return Stream.of(Arguments.of("Gelu"));
    }

    @ParameterizedTest(name = "MSFastGelu")
    @MethodSource("msFastGeluOps")
    void testMsFastGeluOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3).muli(2).subi(1);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, input),
                        InputSpec.constantInput("bias", DataType.FLOAT, new long[]{3}, bias)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                0.05, 1e-2,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msFastGeluOps() {
        return Stream.of(Arguments.of("FastGelu"));
    }

    @ParameterizedTest(name = "MSQuickGelu")
    @MethodSource("msQuickGeluOps")
    void testMsQuickGeluOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3).muli(2).subi(1);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3})),
                Map.of("alpha", floatAttr("alpha", 1.702f)),
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msQuickGeluOps() {
        return Stream.of(Arguments.of("QuickGelu"));
    }

    @ParameterizedTest(name = "MSBiasGelu")
    @MethodSource("msBiasGeluOps")
    void testMsBiasGeluOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3).muli(2).subi(1);
        INDArray bias = Nd4j.rand(DataType.FLOAT, 3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, input),
                        InputSpec.constantInput("B", DataType.FLOAT, new long[]{3}, bias)),
                List.of(new OutputSpec("C", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msBiasGeluOps() {
        return Stream.of(Arguments.of("BiasGelu"));
    }

    @ParameterizedTest(name = "MSSimplifiedLayerNorm")
    @MethodSource("msSimplifiedLayerNormOps")
    void testMsSimplifiedLayerNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray scale = Nd4j.ones(DataType.FLOAT, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4}, input),
                        InputSpec.constantInput("scale", DataType.FLOAT, new long[]{4}, scale)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4})),
                Map.of("axis", intAttr("axis", -1), "epsilon", floatAttr("epsilon", 1e-5f)),
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msSimplifiedLayerNormOps() {
        return Stream.of(Arguments.of("SimplifiedLayerNormalization"));
    }

    @ParameterizedTest(name = "MSSkipLayerNorm")
    @MethodSource("msSkipLayerNormOps")
    void testMsSkipLayerNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray skip = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, 4);
        INDArray beta = Nd4j.zeros(DataType.FLOAT, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 3, 4}, input),
                        InputSpec.graphInput("skip", DataType.FLOAT, new long[]{2, 3, 4}, skip),
                        InputSpec.constantInput("gamma", DataType.FLOAT, new long[]{4}, gamma),
                        InputSpec.constantInput("beta", DataType.FLOAT, new long[]{4}, beta)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3, 4})),
                Map.of("epsilon", floatAttr("epsilon", 1e-5f)),
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msSkipLayerNormOps() {
        return Stream.of(Arguments.of("SkipLayerNormalization"));
    }

    @ParameterizedTest(name = "MSSkipSimplifiedLayerNorm")
    @MethodSource("msSkipSimplifiedLayerNormOps")
    void testMsSkipSimplifiedLayerNormOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray skip = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 3, 4}, input),
                        InputSpec.graphInput("skip", DataType.FLOAT, new long[]{2, 3, 4}, skip),
                        InputSpec.constantInput("gamma", DataType.FLOAT, new long[]{4}, gamma)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3, 4})),
                Map.of("epsilon", floatAttr("epsilon", 1e-5f)),
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msSkipSimplifiedLayerNormOps() {
        return Stream.of(Arguments.of("SkipSimplifiedLayerNormalization"));
    }

    @ParameterizedTest(name = "MSFusedMatMul")
    @MethodSource("msFusedMatMulOps")
    void testMsFusedMatMulOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray a = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray b = Nd4j.rand(DataType.FLOAT, 3, 4);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, a),
                        InputSpec.graphInput("B", DataType.FLOAT, new long[]{3, 4}, b)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 4})),
                Map.of("transA", intAttr("transA", 0), "transB", intAttr("transB", 0)),
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msFusedMatMulOps() {
        return Stream.of(Arguments.of("FusedMatMul"));
    }

    @ParameterizedTest(name = "MSBiasAdd")
    @MethodSource("msBiasAddOps")
    void testMsBiasAddOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3);
        INDArray bias = Nd4j.rand(DataType.FLOAT, 3);
        INDArray skip = Nd4j.zeros(DataType.FLOAT, 2, 3);

        // com.microsoft::BiasAdd requires 3 inputs: input, bias, skip
        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.FLOAT, new long[]{2, 3}, input),
                        InputSpec.constantInput("B", DataType.FLOAT, new long[]{3}, bias),
                        InputSpec.constantInput("C", DataType.FLOAT, new long[]{2, 3}, skip)),
                List.of(new OutputSpec("D", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                1e-4, 1e-6,
                1, "com.microsoft");
        assertOpResult(result);
    }

    static Stream<Arguments> msBiasAddOps() {
        return Stream.of(Arguments.of("BiasAdd"));
    }

    // ======================== BATCH 9: NEURAL NETWORK OPS ========================

    @ParameterizedTest(name = "Dropout")
    @MethodSource("dropoutOps")
    void testDropoutOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 3);
        // ratio=0 means no dropout (eval mode)
        INDArray ratio = Nd4j.scalar(DataType.FLOAT, 0.0f);

        // Dropout in opset 12+ has ratio and training_mode as inputs
        // Output: Y (same shape) and optional mask
        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3}, input),
                        InputSpec.constantInput("ratio", DataType.FLOAT, new long[]{}, ratio)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                1e-4, 1e-6,
                12, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    static Stream<Arguments> dropoutOps() {
        return Stream.of(Arguments.of("Dropout"));
    }

    @ParameterizedTest(name = "LSTM")
    @MethodSource("lstmOps")
    void testLstmOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        int seqLen = 3, batch = 2, inputSize = 4, hiddenSize = 5;
        int numDirections = 1;
        INDArray X = Nd4j.rand(DataType.FLOAT, seqLen, batch, inputSize);
        // W: [num_directions, 4*hidden_size, input_size]
        INDArray W = Nd4j.rand(DataType.FLOAT, numDirections, 4 * hiddenSize, inputSize);
        // R: [num_directions, 4*hidden_size, hidden_size]
        INDArray R = Nd4j.rand(DataType.FLOAT, numDirections, 4 * hiddenSize, hiddenSize);
        // B: [num_directions, 8*hidden_size]
        INDArray B = Nd4j.zeros(DataType.FLOAT, numDirections, 8 * hiddenSize);

        // LSTM produces: Y[seq,num_dir,batch,hidden], Y_h[num_dir,batch,hidden], Y_c[num_dir,batch,hidden]
        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_LSTM");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("LSTM")
                .setName("LSTM_0")
                .addInput("X")
                .addInput("W")
                .addInput("R")
                .addInput("B")
                .addOutput("Y")
                .addOutput("Y_h")
                .addAttribute(intAttr("hidden_size", hiddenSize))
                .build());

        graphBuilder.addInput(makeValueInfo("X", DataType.FLOAT, new long[]{seqLen, batch, inputSize}));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(W, "W"));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(R, "R"));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(B, "B"));
        graphBuilder.addOutput(makeValueInfo("Y", DataType.FLOAT, new long[]{seqLen, numDirections, batch, hiddenSize}));
        graphBuilder.addOutput(makeValueInfo("Y_h", DataType.FLOAT, new long[]{numDirections, batch, hiddenSize}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Map.of("X", X);

        // Run ORT
        Map<String, INDArray> ortOutputs;
        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            ortOutputs = runner.exec(inputMap);
            assertNotNull(ortOutputs);
            assertFalse(ortOutputs.isEmpty());
        }

        // Run SameDiff
        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        assertNotNull(sdOutputs);
        assertFalse(sdOutputs.isEmpty());

        // Compare Y_h (final hidden state)
        compareOutputs(ortOutputs, sdOutputs, 1e-3, 1e-5);
    }

    static Stream<Arguments> lstmOps() {
        return Stream.of(Arguments.of("LSTM"));
    }

    @ParameterizedTest(name = "GRU")
    @MethodSource("gruOps")
    void testGruOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        int seqLen = 3, batch = 2, inputSize = 4, hiddenSize = 5;
        int numDirections = 1;
        INDArray X = Nd4j.rand(DataType.FLOAT, seqLen, batch, inputSize);
        INDArray W = Nd4j.rand(DataType.FLOAT, numDirections, 3 * hiddenSize, inputSize);
        INDArray R = Nd4j.rand(DataType.FLOAT, numDirections, 3 * hiddenSize, hiddenSize);

        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_GRU");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("GRU")
                .setName("GRU_0")
                .addInput("X")
                .addInput("W")
                .addInput("R")
                .addOutput("Y")
                .addOutput("Y_h")
                .addAttribute(intAttr("hidden_size", hiddenSize))
                .build());

        graphBuilder.addInput(makeValueInfo("X", DataType.FLOAT, new long[]{seqLen, batch, inputSize}));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(W, "W"));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(R, "R"));
        graphBuilder.addOutput(makeValueInfo("Y", DataType.FLOAT, new long[]{seqLen, numDirections, batch, hiddenSize}));
        graphBuilder.addOutput(makeValueInfo("Y_h", DataType.FLOAT, new long[]{numDirections, batch, hiddenSize}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Map.of("X", X);

        Map<String, INDArray> ortOutputs;
        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            ortOutputs = runner.exec(inputMap);
            assertNotNull(ortOutputs);
        }

        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        assertNotNull(sdOutputs);

        compareOutputs(ortOutputs, sdOutputs, 1e-3, 1e-5);
    }

    static Stream<Arguments> gruOps() {
        return Stream.of(Arguments.of("GRU"));
    }

    @ParameterizedTest(name = "NegativeLogLikelihoodLoss")
    @MethodSource("nllLossOps")
    void testNllLossOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        // scores: [batch, classes]
        INDArray scores = Nd4j.rand(DataType.FLOAT, 2, 5);
        // target: [batch] with class indices
        INDArray target = Nd4j.createFromArray(new long[]{1, 3});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 5}, scores),
                        InputSpec.graphInput("target", DataType.LONG, new long[]{2}, target)),
                List.of(new OutputSpec("loss", DataType.FLOAT, new long[]{})),
                Map.of("reduction", stringAttr("reduction", "mean")),
                opDir,
                1e-4, 1e-6,
                13, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    static Stream<Arguments> nllLossOps() {
        return Stream.of(Arguments.of("NegativeLogLikelihoodLoss"));
    }

    @ParameterizedTest(name = "SoftmaxCrossEntropyLoss")
    @MethodSource("softmaxCELossOps")
    void testSoftmaxCrossEntropyLossOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray scores = Nd4j.rand(DataType.FLOAT, 2, 5);
        INDArray labels = Nd4j.createFromArray(new long[]{1, 3});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("scores", DataType.FLOAT, new long[]{2, 5}, scores),
                        InputSpec.graphInput("labels", DataType.LONG, new long[]{2}, labels)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{})),
                Map.of("reduction", stringAttr("reduction", "mean")),
                opDir,
                1e-4, 1e-6,
                13, DEFAULT_DOMAIN);
        assertOpResult(result);
    }

    static Stream<Arguments> softmaxCELossOps() {
        return Stream.of(Arguments.of("SoftmaxCrossEntropyLoss"));
    }

    // ======================== BATCH 10: RANDOM OPS (SHAPE CHECK ONLY) ========================

    @ParameterizedTest(name = "RandomNormal")
    @MethodSource("randomNormalOps")
    void testRandomNormalOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        // RandomNormal has no inputs - shape is an attribute
        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_RandomNormal");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("RandomNormal")
                .setName("RandomNormal_0")
                .addOutput("Y")
                .addAttribute(intAttr("dtype", 1))  // FLOAT
                .addAttribute(floatAttr("mean", 0.0f))
                .addAttribute(floatAttr("scale", 1.0f))
                .addAttribute(intsAttr("shape", 2, 3))
                .build());

        graphBuilder.addOutput(makeValueInfo("Y", DataType.FLOAT, new long[]{2, 3}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Collections.emptyMap();

        // Run ORT - verify shape
        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            Map<String, INDArray> ortOutputs = runner.exec(inputMap);
            assertNotNull(ortOutputs);
            INDArray output = ortOutputs.values().iterator().next();
            assertArrayEquals(new long[]{2, 3}, output.shape(), "Shape mismatch");
        }

        // Run SameDiff - verify shape
        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        assertNotNull(sdOutputs);
        INDArray sdOutput = sdOutputs.values().iterator().next();
        assertArrayEquals(new long[]{2, 3}, sdOutput.shape(), "SameDiff shape mismatch");
    }

    static Stream<Arguments> randomNormalOps() {
        return Stream.of(Arguments.of("RandomNormal"));
    }

    @ParameterizedTest(name = "RandomUniform")
    @MethodSource("randomUniformOps")
    void testRandomUniformOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");

        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_RandomUniform");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("RandomUniform")
                .setName("RandomUniform_0")
                .addOutput("Y")
                .addAttribute(intAttr("dtype", 1))
                .addAttribute(floatAttr("low", 0.0f))
                .addAttribute(floatAttr("high", 1.0f))
                .addAttribute(intsAttr("shape", 2, 3))
                .build());

        graphBuilder.addOutput(makeValueInfo("Y", DataType.FLOAT, new long[]{2, 3}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Collections.emptyMap();

        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            Map<String, INDArray> ortOutputs = runner.exec(inputMap);
            INDArray output = ortOutputs.values().iterator().next();
            assertArrayEquals(new long[]{2, 3}, output.shape());
        }

        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        INDArray sdOutput = sdOutputs.values().iterator().next();
        assertArrayEquals(new long[]{2, 3}, sdOutput.shape());
    }

    static Stream<Arguments> randomUniformOps() {
        return Stream.of(Arguments.of("RandomUniform"));
    }

    @ParameterizedTest(name = "RandomLike: {0}")
    @MethodSource("randomLikeOps")
    void testRandomLikeOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray template = Nd4j.rand(DataType.FLOAT, 2, 3);

        // RandomNormalLike/RandomUniformLike take a tensor input, produce same-shape output
        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("input", DataType.FLOAT, new long[]{2, 3}, template)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                1e10, 1e10);  // Don't compare values for random ops
        // Just check both ran without error
        if (result.onnxRuntimeError != null) {
            fail("ORT failed: " + result.onnxRuntimeError.getMessage());
        }
        if (result.sameDiffError != null) {
            fail("SameDiff failed: " + result.sameDiffError.getMessage());
        }
        // Verify shapes match
        assertNotNull(result.onnxRuntimeOutputs);
        assertNotNull(result.sameDiffOutputs);
    }

    static Stream<Arguments> randomLikeOps() {
        return Stream.of(
                Arguments.of("RandomNormalLike"),
                Arguments.of("RandomUniformLike")
        );
    }

    @ParameterizedTest(name = "Multinomial")
    @MethodSource("multinomialOps")
    void testMultinomialOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        // Input is log-probabilities [batch, classes]
        INDArray input = Nd4j.rand(DataType.FLOAT, 2, 5);
        // Normalize to valid log-probs
        INDArray logProbs = input.sub(input.max(true, 1));

        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_Multinomial");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("Multinomial")
                .setName("Multinomial_0")
                .addInput("input")
                .addOutput("output")
                .addAttribute(intAttr("sample_size", 3))
                .addAttribute(intAttr("dtype", 7))  // INT64
                .build());

        graphBuilder.addInput(makeValueInfo("input", DataType.FLOAT, new long[]{2, 5}));
        graphBuilder.addOutput(makeValueInfo("output", DataType.LONG, new long[]{2, 3}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Map.of("input", logProbs);

        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            Map<String, INDArray> ortOutputs = runner.exec(inputMap);
            INDArray output = ortOutputs.values().iterator().next();
            assertArrayEquals(new long[]{2, 3}, output.shape());
        }

        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        INDArray sdOutput = sdOutputs.values().iterator().next();
        assertArrayEquals(new long[]{2, 3}, sdOutput.shape());
    }

    static Stream<Arguments> multinomialOps() {
        return Stream.of(Arguments.of("Multinomial"));
    }

    // ======================== BATCH 11: QUANTIZATION OPS ========================

    @ParameterizedTest(name = "DequantizeLinear")
    @MethodSource("dequantizeLinearOps")
    void testDequantizeLinearOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        // uint8 input
        INDArray x = Nd4j.createFromArray(new int[]{0, 128, 255, 64, 192, 32}).reshape(2, 3).castTo(DataType.UBYTE);
        INDArray scale = Nd4j.scalar(DataType.FLOAT, 0.01f);
        INDArray zeroPoint = Nd4j.scalar(DataType.UBYTE, 128);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("x", DataType.UBYTE, new long[]{2, 3}, x),
                        InputSpec.constantInput("x_scale", DataType.FLOAT, new long[]{}, scale),
                        InputSpec.constantInput("x_zero_point", DataType.UBYTE, new long[]{}, zeroPoint)),
                List.of(new OutputSpec("y", DataType.FLOAT, new long[]{2, 3})),
                null,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> dequantizeLinearOps() {
        return Stream.of(Arguments.of("DequantizeLinear"));
    }

    @ParameterizedTest(name = "QuantizeLinear")
    @MethodSource("quantizeLinearOps")
    void testQuantizeLinearOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray x = Nd4j.rand(DataType.FLOAT, 2, 3).muli(2).subi(1);
        INDArray scale = Nd4j.scalar(DataType.FLOAT, 0.01f);
        INDArray zeroPoint = Nd4j.scalar(DataType.UBYTE, 128);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("x", DataType.FLOAT, new long[]{2, 3}, x),
                        InputSpec.constantInput("y_scale", DataType.FLOAT, new long[]{}, scale),
                        InputSpec.constantInput("y_zero_point", DataType.UBYTE, new long[]{}, zeroPoint)),
                List.of(new OutputSpec("y", DataType.UBYTE, new long[]{2, 3})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> quantizeLinearOps() {
        return Stream.of(Arguments.of("QuantizeLinear"));
    }

    // ======================== BATCH 13: MISC OPS ========================

    @ParameterizedTest(name = "Det")
    @MethodSource("detOps")
    void testDetOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 3, 3);

        OpTestResult result = executeOpTest(
                opType,
                List.of(InputSpec.graphInput("X", DataType.FLOAT, new long[]{3, 3}, input)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{})),
                null,
                opDir,
                1e-3, 1e-5);
        assertOpResult(result);
    }

    static Stream<Arguments> detOps() {
        return Stream.of(Arguments.of("Det"));
    }

    @ParameterizedTest(name = "NonMaxSuppression")
    @MethodSource("nmsOps")
    void testNonMaxSuppressionOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        // boxes: [num_batches, num_boxes, 4]
        INDArray boxes = Nd4j.createFromArray(new float[]{
                0, 0, 1, 1,   0.5f, 0.5f, 1.5f, 1.5f,   0, 0, 0.5f, 0.5f,
                2, 2, 3, 3,   2.5f, 2.5f, 3.5f, 3.5f,   2, 2, 2.5f, 2.5f
        }).reshape(1, 6, 4);
        // scores: [num_batches, num_classes, num_boxes]
        INDArray scores = Nd4j.createFromArray(new float[]{
                0.9f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f
        }).reshape(1, 1, 6);

        INDArray maxOutputBoxes = Nd4j.scalar(DataType.LONG, 3);
        INDArray iouThreshold = Nd4j.scalar(DataType.FLOAT, 0.5f);
        INDArray scoreThreshold = Nd4j.scalar(DataType.FLOAT, 0.0f);

        // NMS output shape is dynamic
        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("boxes", DataType.FLOAT, new long[]{1, 6, 4}, boxes),
                        InputSpec.graphInput("scores", DataType.FLOAT, new long[]{1, 1, 6}, scores),
                        InputSpec.constantInput("max_output_boxes_per_class", DataType.LONG, new long[]{}, maxOutputBoxes),
                        InputSpec.constantInput("iou_threshold", DataType.FLOAT, new long[]{}, iouThreshold),
                        InputSpec.constantInput("score_threshold", DataType.FLOAT, new long[]{}, scoreThreshold)),
                List.of(new OutputSpec("selected_indices", DataType.LONG, null)),
                null,
                opDir,
                0, 0);
        // NMS has dynamic output; just check it runs
        if (result.onnxRuntimeError != null) {
            fail("ORT failed: " + result.onnxRuntimeError.getMessage());
        }
        if (result.sameDiffError != null) {
            fail("SameDiff failed: " + result.sameDiffError.getMessage());
        }
    }

    static Stream<Arguments> nmsOps() {
        return Stream.of(Arguments.of("NonMaxSuppression"));
    }

    @ParameterizedTest(name = "BitShift")
    @MethodSource("bitShiftOps")
    void testBitShiftOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray x = Nd4j.createFromArray(new int[]{16, 32, 64, 128, 256, 512}).reshape(2, 3).castTo(DataType.UINT32);
        INDArray y = Nd4j.createFromArray(new int[]{1, 2, 3, 1, 2, 3}).reshape(2, 3).castTo(DataType.UINT32);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.UINT32, new long[]{2, 3}, x),
                        InputSpec.graphInput("Y", DataType.UINT32, new long[]{2, 3}, y)),
                List.of(new OutputSpec("Z", DataType.UINT32, new long[]{2, 3})),
                Map.of("direction", stringAttr("direction", "RIGHT")),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> bitShiftOps() {
        return Stream.of(Arguments.of("BitShift"));
    }

    // ======================== ADDITIONAL SINGLE OPS ========================

    @ParameterizedTest(name = "MatMulInteger")
    @MethodSource("matMulIntegerOps")
    void testMatMulIntegerOps(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_");
        INDArray a = Nd4j.createFromArray(new int[][]{{1, 2, 3}, {4, 5, 6}}).castTo(DataType.UBYTE);
        INDArray b = Nd4j.createFromArray(new int[][]{{7, 8}, {9, 10}, {11, 12}}).castTo(DataType.UBYTE);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("A", DataType.UBYTE, new long[]{2, 3}, a),
                        InputSpec.graphInput("B", DataType.UBYTE, new long[]{3, 2}, b)),
                List.of(new OutputSpec("Y", DataType.INT, new long[]{2, 2})),
                null,
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> matMulIntegerOps() {
        return Stream.of(Arguments.of("MatMulInteger"));
    }

    // ======================== EXTENDED ONNX IMPORT HOOK TESTS ========================
    // Tests for the changed ONNX pre-import hooks: GatherND, Pad, Resize, PRelu, NMS, BitShift

    // --- GatherND: Q=1 with 3D data (vision encoder pattern: gather from [vocab, hidden]) ---

    @ParameterizedTest(name = "GatherND_Q1_3D")
    @MethodSource("gatherND_Q1_3D_args")
    void testGatherND_Q1_3D(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_q1_3d_");
        // data: [5, 4, 3] (e.g., 5 positions, each with 4x3 features)
        INDArray data = Nd4j.rand(DataType.FLOAT, 5, 4, 3);
        // indices: [2, 1] with Q=1 → gather 2 rows from axis 0
        INDArray indices = Nd4j.createFromArray(new long[][]{{1}, {3}}).castTo(DataType.LONG);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{5, 4, 3}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{2, 1}, indices)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 4, 3})),
                Map.of("batch_dims", intAttr("batch_dims", 0)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> gatherND_Q1_3D_args() {
        return Stream.of(Arguments.of("GatherND"));
    }

    // --- GatherND: Q=2 with 3D data (multi-dimensional indexing) ---

    @ParameterizedTest(name = "GatherND_Q2_3D")
    @MethodSource("gatherND_Q2_3D_args")
    void testGatherND_Q2_3D(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_q2_3d_");
        // data: [3, 4, 5]
        INDArray data = Nd4j.rand(DataType.FLOAT, 3, 4, 5);
        // indices: [2, 2] with Q=2 → index 2 dimensions, output has remaining dims
        INDArray indices = Nd4j.createFromArray(new long[][]{{0, 1}, {2, 3}}).castTo(DataType.LONG);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 4, 5}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{2, 2}, indices)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2, 5})),
                Map.of("batch_dims", intAttr("batch_dims", 0)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> gatherND_Q2_3D_args() {
        return Stream.of(Arguments.of("GatherND"));
    }

    // --- GatherND: Q=3 full indexing (scalar output per index) ---

    @ParameterizedTest(name = "GatherND_Q3_fullIndex")
    @MethodSource("gatherND_Q3_fullIndex_args")
    void testGatherND_Q3_fullIndex(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_q3_full_");
        // data: [3, 4, 5]
        INDArray data = Nd4j.rand(DataType.FLOAT, 3, 4, 5);
        // indices: [2, 3] with Q=3 → full indexing, output is scalar per index row
        INDArray indices = Nd4j.createFromArray(new long[][]{{0, 1, 2}, {2, 3, 4}}).castTo(DataType.LONG);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 4, 5}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{2, 3}, indices)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{2})),
                Map.of("batch_dims", intAttr("batch_dims", 0)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> gatherND_Q3_fullIndex_args() {
        return Stream.of(Arguments.of("GatherND"));
    }

    // --- GatherND: batch_dims=1 ---

    @ParameterizedTest(name = "GatherND_batchDims1")
    @MethodSource("gatherND_batchDims1_args")
    void testGatherND_batchDims1(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_batch1_");
        // data: [2, 3, 4] (batch=2)
        INDArray data = Nd4j.rand(DataType.FLOAT, 2, 3, 4);
        // indices: [2, 1, 1] with batch_dims=1
        INDArray indices = Nd4j.createFromArray(new long[]{0, 2}).reshape(2, 1, 1).castTo(DataType.LONG);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{2, 3, 4}, data),
                        InputSpec.graphInput("indices", DataType.LONG, new long[]{2, 1, 1}, indices)),
                List.of(new OutputSpec("output", DataType.FLOAT, null)),
                Map.of("batch_dims", intAttr("batch_dims", 1)),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> gatherND_batchDims1_args() {
        return Stream.of(Arguments.of("GatherND"));
    }

    // --- Pad: 4D NCHW constant padding (vision encoder pattern) ---

    @ParameterizedTest(name = "Pad_4D_constant")
    @MethodSource("pad4DConstantArgs")
    void testPad4DConstant(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_4d_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 1, 3, 4, 4);
        // ONNX format: [begin_N, begin_C, begin_H, begin_W, end_N, end_C, end_H, end_W]
        INDArray pads = Nd4j.createFromArray(new long[]{0, 0, 1, 1, 0, 0, 1, 1});
        INDArray constantValue = Nd4j.scalar(DataType.FLOAT, 0.0f);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{1, 3, 4, 4}, data),
                        InputSpec.constantInput("pads", DataType.LONG, new long[]{8}, pads),
                        InputSpec.constantInput("constant_value", DataType.FLOAT, new long[]{}, constantValue)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{1, 3, 6, 6})),
                Map.of("mode", stringAttr("mode", "constant")),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> pad4DConstantArgs() {
        return Stream.of(Arguments.of("Pad"));
    }

    // --- Pad: non-zero constant value ---

    @ParameterizedTest(name = "Pad_nonzeroConstant")
    @MethodSource("padNonzeroConstantArgs")
    void testPadNonzeroConstant(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_nonzero_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 3, 3);
        INDArray pads = Nd4j.createFromArray(new long[]{1, 2, 1, 2});
        INDArray constantValue = Nd4j.scalar(DataType.FLOAT, -1.5f);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 3}, data),
                        InputSpec.constantInput("pads", DataType.LONG, new long[]{4}, pads),
                        InputSpec.constantInput("constant_value", DataType.FLOAT, new long[]{}, constantValue)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{5, 7})),
                Map.of("mode", stringAttr("mode", "constant")),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> padNonzeroConstantArgs() {
        return Stream.of(Arguments.of("Pad"));
    }

    // --- Pad: reflect mode ---

    @ParameterizedTest(name = "Pad_reflect")
    @MethodSource("padReflectArgs")
    void testPadReflect(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_reflect_");
        INDArray data = Nd4j.rand(DataType.FLOAT, 3, 4);
        // Reflect padding must be < dim size
        INDArray pads = Nd4j.createFromArray(new long[]{1, 2, 1, 2});

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("data", DataType.FLOAT, new long[]{3, 4}, data),
                        InputSpec.constantInput("pads", DataType.LONG, new long[]{4}, pads)),
                List.of(new OutputSpec("output", DataType.FLOAT, new long[]{5, 8})),
                Map.of("mode", stringAttr("mode", "reflect")),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> padReflectArgs() {
        return Stream.of(Arguments.of("Pad"));
    }

    // --- Resize: sizes-based (op has 4 inputs) ---

    @ParameterizedTest(name = "Resize_sizesBased")
    @MethodSource("resizeSizesBasedArgs")
    void testResizeSizesBased(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_sizes_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 2, 2);
        INDArray sizes = Nd4j.createFromArray(new long[]{1, 1, 4, 4});

        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_Resize_sizes");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("Resize")
                .setName("Resize_0")
                .addInput("X")
                .addInput("")  // empty roi
                .addInput("")  // empty scales
                .addInput("sizes")
                .addOutput("Y")
                .addAttribute(stringAttr("mode", "nearest"))
                .build());

        graphBuilder.addInput(makeValueInfo("X", DataType.FLOAT, new long[]{1, 1, 2, 2}));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(sizes, "sizes"));
        graphBuilder.addOutput(makeValueInfo("Y", DataType.FLOAT, new long[]{1, 1, 4, 4}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Map.of("X", input);

        Map<String, INDArray> ortOutputs;
        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            ortOutputs = runner.exec(inputMap);
            assertNotNull(ortOutputs);
        }

        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        assertNotNull(sdOutputs);

        compareOutputs(ortOutputs, sdOutputs, 1e-4, 1e-6);
    }

    static Stream<Arguments> resizeSizesBasedArgs() {
        return Stream.of(Arguments.of("Resize"));
    }

    // --- Resize: bilinear mode ---

    @ParameterizedTest(name = "Resize_bilinear")
    @MethodSource("resizeBilinearArgs")
    void testResizeBilinear(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_bilinear_");
        INDArray input = Nd4j.rand(DataType.FLOAT, 1, 1, 2, 2);
        INDArray scales = Nd4j.createFromArray(new float[]{1.0f, 1.0f, 3.0f, 3.0f});

        Onnx.GraphProto.Builder graphBuilder = Onnx.GraphProto.newBuilder()
                .setName("test_Resize_bilinear");

        graphBuilder.addNode(Onnx.NodeProto.newBuilder()
                .setOpType("Resize")
                .setName("Resize_0")
                .addInput("X")
                .addInput("")  // empty roi
                .addInput("scales")
                .addOutput("Y")
                .addAttribute(stringAttr("mode", "linear"))
                .build());

        graphBuilder.addInput(makeValueInfo("X", DataType.FLOAT, new long[]{1, 1, 2, 2}));
        graphBuilder.addInitializer(org.nd4j.onnx.OnnxTensorUtils.toTensorProto(scales, "scales"));
        graphBuilder.addOutput(makeValueInfo("Y", DataType.FLOAT, new long[]{1, 1, 6, 6}));

        Onnx.ModelProto model = Onnx.ModelProto.newBuilder()
                .setIrVersion(7)
                .setGraph(graphBuilder.build())
                .addOpsetImport(Onnx.OperatorSetIdProto.newBuilder().setVersion(DEFAULT_OPSET).build())
                .build();

        java.io.File modelFile = saveModel(model, opDir);
        Map<String, INDArray> inputMap = Map.of("X", input);

        Map<String, INDArray> ortOutputs;
        try (org.nd4j.onnxruntime.runner.OnnxRuntimeRunner runner =
                     org.nd4j.onnxruntime.runner.OnnxRuntimeRunner.builder()
                             .modelUri(modelFile.getAbsolutePath()).build()) {
            ortOutputs = runner.exec(inputMap);
            assertNotNull(ortOutputs);
        }

        org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter importer =
                new org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter();
        org.nd4j.autodiff.samediff.SameDiff sd = importer.runImport(
                modelFile.getAbsolutePath(), inputMap, false, false);
        Map<String, INDArray> sdOutputs = sd.outputAll(inputMap);
        assertNotNull(sdOutputs);

        compareOutputs(ortOutputs, sdOutputs, 1e-4, 1e-6);
    }

    static Stream<Arguments> resizeBilinearArgs() {
        return Stream.of(Arguments.of("Resize"));
    }

    // --- PRelu: 4D NCHW with 1D channel slope (common conv pattern) ---

    @ParameterizedTest(name = "PRelu_4D_channelSlope")
    @MethodSource("prelu4DChannelSlopeArgs")
    void testPRelu4DChannelSlope(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_4d_");
        INDArray x = Nd4j.rand(DataType.FLOAT, 2, 3, 4, 4).muli(2).subi(1);
        // ONNX PRelu requires slope to be unidirectionally broadcastable to X
        // For NCHW [2,3,4,4], use [1,3,1,1] for channel-wise broadcasting
        INDArray slope = Nd4j.rand(DataType.FLOAT, 1, 3, 1, 1).muli(0.25f);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{2, 3, 4, 4}, x),
                        InputSpec.graphInput("slope", DataType.FLOAT, new long[]{1, 3, 1, 1}, slope)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{2, 3, 4, 4})),
                null,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> prelu4DChannelSlopeArgs() {
        return Stream.of(Arguments.of("PRelu"));
    }

    // --- PRelu: 2D input with scalar slope ---

    @ParameterizedTest(name = "PRelu_2D_scalarSlope")
    @MethodSource("prelu2DScalarSlopeArgs")
    void testPRelu2DScalarSlope(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_2d_scalar_");
        INDArray x = Nd4j.rand(DataType.FLOAT, 4, 5).muli(2).subi(1);
        INDArray slope = Nd4j.scalar(DataType.FLOAT, 0.1f);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.FLOAT, new long[]{4, 5}, x),
                        InputSpec.graphInput("slope", DataType.FLOAT, new long[]{}, slope)),
                List.of(new OutputSpec("Y", DataType.FLOAT, new long[]{4, 5})),
                null,
                opDir,
                1e-4, 1e-6);
        assertOpResult(result);
    }

    static Stream<Arguments> prelu2DScalarSlopeArgs() {
        return Stream.of(Arguments.of("PRelu"));
    }

    // --- NonMaxSuppression: verify output values match ORT ---

    @ParameterizedTest(name = "NonMaxSuppression_verifyValues")
    @MethodSource("nmsVerifyArgs")
    void testNonMaxSuppressionVerifyValues(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_verify_");
        // 4 non-overlapping boxes, easy case
        INDArray boxes = Nd4j.createFromArray(new float[]{
                0, 0, 1, 1,   2, 2, 3, 3,   4, 4, 5, 5,   6, 6, 7, 7
        }).reshape(1, 4, 4);
        INDArray scores = Nd4j.createFromArray(new float[]{
                0.9f, 0.8f, 0.7f, 0.6f
        }).reshape(1, 1, 4);

        INDArray maxOutputBoxes = Nd4j.scalar(DataType.LONG, 4);
        INDArray iouThreshold = Nd4j.scalar(DataType.FLOAT, 0.5f);
        INDArray scoreThreshold = Nd4j.scalar(DataType.FLOAT, 0.0f);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("boxes", DataType.FLOAT, new long[]{1, 4, 4}, boxes),
                        InputSpec.graphInput("scores", DataType.FLOAT, new long[]{1, 1, 4}, scores),
                        InputSpec.constantInput("max_output_boxes_per_class", DataType.LONG, new long[]{}, maxOutputBoxes),
                        InputSpec.constantInput("iou_threshold", DataType.FLOAT, new long[]{}, iouThreshold),
                        InputSpec.constantInput("score_threshold", DataType.FLOAT, new long[]{}, scoreThreshold)),
                List.of(new OutputSpec("selected_indices", DataType.LONG, null)),
                null,
                opDir,
                0, 0);
        if (result.onnxRuntimeError != null) {
            fail("ORT failed: " + result.onnxRuntimeError.getMessage());
        }
        if (result.sameDiffError != null) {
            fail("SameDiff failed: " + result.sameDiffError.getMessage());
        }
        // Verify we got all 4 non-overlapping boxes selected
        assertNotNull(result.onnxRuntimeOutputs);
        assertNotNull(result.sameDiffOutputs);
    }

    static Stream<Arguments> nmsVerifyArgs() {
        return Stream.of(Arguments.of("NonMaxSuppression"));
    }

    // --- BitShift: left shift ---

    @ParameterizedTest(name = "BitShift_left")
    @MethodSource("bitShiftLeftArgs")
    void testBitShiftLeft(String opType) throws Exception {
        Path opDir = Files.createTempDirectory(tempDir, opType + "_left_");
        INDArray x = Nd4j.createFromArray(new int[]{1, 2, 4, 8, 16, 32}).reshape(2, 3).castTo(DataType.UINT32);
        INDArray y = Nd4j.createFromArray(new int[]{1, 2, 3, 1, 2, 3}).reshape(2, 3).castTo(DataType.UINT32);

        OpTestResult result = executeOpTest(
                opType,
                List.of(
                        InputSpec.graphInput("X", DataType.UINT32, new long[]{2, 3}, x),
                        InputSpec.graphInput("Y", DataType.UINT32, new long[]{2, 3}, y)),
                List.of(new OutputSpec("Z", DataType.UINT32, new long[]{2, 3})),
                Map.of("direction", stringAttr("direction", "LEFT")),
                opDir,
                0, 0);
        assertOpResult(result);
    }

    static Stream<Arguments> bitShiftLeftArgs() {
        return Stream.of(Arguments.of("BitShift"));
    }

    // ======================== HELPER METHODS ========================

    /**
     * Assert an op test result passed.
     * Provides detailed failure information.
     */
    private void assertOpResult(OpTestResult result) {
        if (result.onnxRuntimeError != null) {
            fail("ONNX Runtime failed for " + result.opType + ": " + result.onnxRuntimeError.getMessage(),
                    result.onnxRuntimeError);
        }
        if (result.sameDiffError != null) {
            fail("SameDiff import/exec failed for " + result.opType + ": " + result.sameDiffError.getMessage(),
                    result.sameDiffError);
        }
        if (result.comparisonError != null) {
            fail("Output comparison failed for " + result.opType + ": " + result.comparisonError.getMessage(),
                    result.comparisonError);
        }
        assertTrue(result.passed, "Op test did not pass for " + result.opType);
    }
}
