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
package org.nd4j.onnxruntime.runner;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.onnxruntime.*;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.onnx.OnnxTensorUtils;
import org.nd4j.onnxruntime.runner.enums.ONNXType;
import org.nd4j.onnxruntime.util.ONNXUtils;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;
import static org.nd4j.onnxruntime.util.ONNXUtils.*;

@Slf4j
@Getter
public class OnnxRuntimeRunner implements Closeable {

    private Session session;
    private RunOptions runOptions;
    private MemoryInfo memoryInfo;
    private OrtAllocator allocator;
    private SessionOptions sessionOptions;
    private static Env env;
    private Pointer bp;
    private Onnx.ModelProto modelProto;

    // Store the original model path for reloading
    private final String originalModelUri;
    private String currentModelPath; // Path to currently loaded model (may be modified)

    // Track current session configuration
    private List<String> currentOutputNames;
    private boolean sessionNeedsReload = false;

    @Getter
    private List<Onnx.TensorProto> initializers = new ArrayList<>();
    @Getter
    private List<Onnx.ValueInfoProto> inputs = new ArrayList<>();
    @Getter
    private List<Onnx.ValueInfoProto> outputs = new ArrayList<>();

    // Cache of all possible output names and their types from the model graph
    private Map<String, Onnx.ValueInfoProto> allAvailableOutputs;

    @Builder
    public OnnxRuntimeRunner(String modelUri) {
        this.originalModelUri = modelUri;
        this.currentModelPath = modelUri;

        if (env == null) {
            // First create a basic environment to register the default logger
            env = new Env(ONNXUtils.getOnnxLogLevelFromLogger(log),
                    new BytePointer("nd4j-serving-onnx-session-" + UUID.randomUUID()));
            env.retainReference();
        }

        initializeSessionOptions();

        allocator = new OrtAllocator();
        allocator.retainReference();

        if (modelUri != null) {
            loadModel();
            createSession();
        } else {
            runOptions = new RunOptions();
            memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        }

        // Initialize with default outputs
        this.currentOutputNames = getDefaultOutputNames();
    }

    /**
     * Initialize session options with default configuration
     */
    private void initializeSessionOptions() {
        sessionOptions = new SessionOptions();
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.SetExecutionMode(ORT_SEQUENTIAL);
        sessionOptions.EnableCpuMemArena();
        sessionOptions.EnableMemPattern();
        sessionOptions.AddConfigEntry(new BytePointer("ep.context_enable"), new BytePointer("1"));
        sessionOptions.AddConfigEntry(new BytePointer("session.use_env_allocators"), new BytePointer("1"));
        sessionOptions.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
        sessionOptions.retainReference();
    }

    /**
     * Load the ONNX model proto and extract metadata
     */
    private void loadModel() {
        try {
            modelProto = Onnx.ModelProto.parseFrom(FileUtils.readFileToByteArray(new File(currentModelPath)));
        } catch (IOException e) {
            log.error("Failed to parse model proto", e);
            throw new RuntimeException("Failed to load model: " + currentModelPath, e);
        }

        // Clear and reload metadata
        initializers.clear();
        inputs.clear();
        outputs.clear();

        for (int i = 0; i < modelProto.getGraph().getInitializerCount(); i++) {
            initializers.add(modelProto.getGraph().getInitializer(i));
        }
        for (int i = 0; i < modelProto.getGraph().getInputCount(); i++) {
            inputs.add(modelProto.getGraph().getInput(i));
        }
        for (int i = 0; i < modelProto.getGraph().getOutputCount(); i++) {
            outputs.add(modelProto.getGraph().getOutput(i));
        }

        // Build cache of all available outputs with their ACTUAL types
        allAvailableOutputs = new HashMap<>();

        // Add existing graph outputs
        for (Onnx.ValueInfoProto output : outputs) {
            allAvailableOutputs.put(output.getName(), output);
        }

        // Create maps for easy lookup
        Map<String, Onnx.ValueInfoProto> valueInfoMap = new HashMap<>();
        for (Onnx.ValueInfoProto valueInfo : modelProto.getGraph().getValueInfoList()) {
            valueInfoMap.put(valueInfo.getName(), valueInfo);
        }

        Map<String, Onnx.TensorProto> initializerMap = new HashMap<>();
        for (Onnx.TensorProto initializer : modelProto.getGraph().getInitializerList()) {
            initializerMap.put(initializer.getName(), initializer);
        }

        Map<String, Onnx.ValueInfoProto> inputMap = new HashMap<>();
        for (Onnx.ValueInfoProto input : modelProto.getGraph().getInputList()) {
            inputMap.put(input.getName(), input);
        }

        // Process all node outputs with improved type inference
        processAllNodeOutputs(valueInfoMap, initializerMap, inputMap);
    }

    /**
     * Process all node outputs and ensure they're all added to available outputs
     */
    private void processAllNodeOutputs(Map<String, Onnx.ValueInfoProto> valueInfoMap,
                                       Map<String, Onnx.TensorProto> initializerMap,
                                       Map<String, Onnx.ValueInfoProto> inputMap) {

        // Build a comprehensive type resolution map using multiple passes
        Map<String, Onnx.ValueInfoProto> resolvedTypes = new HashMap<>();

        // First pass: Add all known types
        resolvedTypes.putAll(valueInfoMap);
        resolvedTypes.putAll(inputMap);

        // Add initializers as value info
        for (Map.Entry<String, Onnx.TensorProto> entry : initializerMap.entrySet()) {
            if (!resolvedTypes.containsKey(entry.getKey())) {
                resolvedTypes.put(entry.getKey(), createValueInfoFromInitializer(entry.getValue()));
            }
        }

        // Multiple passes to resolve all node outputs
        boolean typesResolved = true;
        int maxPasses = 10; // Prevent infinite loops
        int passCount = 0;

        do {
            typesResolved = true;
            passCount++;

            for (Onnx.NodeProto node : modelProto.getGraph().getNodeList()) {
                for (String nodeOutput : node.getOutputList()) {
                    if (!resolvedTypes.containsKey(nodeOutput) && !allAvailableOutputs.containsKey(nodeOutput)) {
                        Onnx.ValueInfoProto inferredType = inferOutputTypeImproved(node, nodeOutput, resolvedTypes);
                        if (inferredType != null) {
                            resolvedTypes.put(nodeOutput, inferredType);
                            allAvailableOutputs.put(nodeOutput, inferredType);
                        } else {
                            // Create a fallback type if we can't infer - ensures ALL outputs are available
                            Onnx.ValueInfoProto fallbackType = createFallbackOutputType(nodeOutput, node.getOpType());
                            resolvedTypes.put(nodeOutput, fallbackType);
                            allAvailableOutputs.put(nodeOutput, fallbackType);
                            log.warn("Using fallback type for output '{}' from node '{}' (op: {})",
                                    nodeOutput, node.getName(), node.getOpType());
                        }
                    } else if (resolvedTypes.containsKey(nodeOutput) && !allAvailableOutputs.containsKey(nodeOutput)) {
                        // Add to available outputs if resolved but not added yet
                        allAvailableOutputs.put(nodeOutput, resolvedTypes.get(nodeOutput));
                    }
                }
            }
        } while (!typesResolved && passCount < maxPasses);

        log.debug("Resolved all node outputs in {} passes. Total available outputs: {}",
                passCount, allAvailableOutputs.size());
    }

    /**
     * Create ValueInfoProto from TensorProto initializer
     */
    private Onnx.ValueInfoProto createValueInfoFromInitializer(Onnx.TensorProto initializer) {
        Onnx.ValueInfoProto.Builder builder = Onnx.ValueInfoProto.newBuilder();
        builder.setName(initializer.getName());

        Onnx.TypeProto.Tensor.Builder tensorBuilder = Onnx.TypeProto.Tensor.newBuilder();
        tensorBuilder.setElemType(initializer.getDataType());

        // Add shape if available
        if (initializer.getDimsCount() > 0) {
            Onnx.TensorShapeProto.Builder shapeBuilder = Onnx.TensorShapeProto.newBuilder();
            for (long dim : initializer.getDimsList()) {
                Onnx.TensorShapeProto.Dimension.Builder dimBuilder =
                        Onnx.TensorShapeProto.Dimension.newBuilder();
                dimBuilder.setDimValue(dim);
                shapeBuilder.addDim(dimBuilder.build());
            }
            tensorBuilder.setShape(shapeBuilder.build());
        }

        Onnx.TypeProto.Builder typeBuilder = Onnx.TypeProto.newBuilder();
        typeBuilder.setTensorType(tensorBuilder.build());
        builder.setType(typeBuilder.build());

        return builder.build();
    }

    /**
     * Improved output type inference that handles more cases and uses resolved types
     */
    private Onnx.ValueInfoProto inferOutputTypeImproved(Onnx.NodeProto node, String outputName,
                                                        Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        String opType = node.getOpType();

        // Handle specific operations with known type transformations
        switch (opType) {
            case "Equal":
            case "Greater":
            case "Less":
            case "GreaterOrEqual":
            case "LessOrEqual":
            case "And":
            case "Or":
            case "Not":
                return createBooleanOutputType(outputName, getShapeFromFirstInput(node, resolvedTypes));

            case "Shape":
                return createInt64OutputType(outputName, null); // Shape output is always 1D

            case "Cast":
                return handleCastOperation(node, outputName, resolvedTypes);

            case "ConstantOfShape":
                return handleConstantOfShapeOperation(node, outputName, resolvedTypes);

            case "Gather":
            case "GatherElements":
            case "GatherND":
                return handleGatherOperation(node, outputName, resolvedTypes);

            case "Expand":
            case "Broadcast":
                return handleExpandOperation(node, outputName, resolvedTypes);

            case "Where":
                return handleWhereOperation(node, outputName, resolvedTypes);

            case "Unsqueeze":
            case "Squeeze":
            case "Reshape":
            case "Transpose":
            case "Flatten":
                return handleShapeManipulationOperation(node, outputName, resolvedTypes);

            case "Slice":
                return handleSliceOperation(node, outputName, resolvedTypes);

            case "Concat":
                return handleConcatOperation(node, outputName, resolvedTypes);

            case "Split":
                return handleSplitOperation(node, outputName, resolvedTypes);

            default:
                // For most operations, try to preserve the first input type
                return inheritFromFirstInput(node, outputName, resolvedTypes);
        }
    }

    /**
     * Handle Cast operation - get target type from 'to' attribute
     */
    private Onnx.ValueInfoProto handleCastOperation(Onnx.NodeProto node, String outputName,
                                                    Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        for (Onnx.AttributeProto attr : node.getAttributeList()) {
            if (attr.getName().equals("to")) {
                int targetType = (int) attr.getI();
                return createTypedOutputType(outputName, targetType, getShapeFromFirstInput(node, resolvedTypes));
            }
        }
        return inheritFromFirstInput(node, outputName, resolvedTypes);
    }

    /**
     * Handle ConstantOfShape operation
     */
    private Onnx.ValueInfoProto handleConstantOfShapeOperation(Onnx.NodeProto node, String outputName,
                                                               Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        // Check value attribute for data type
        for (Onnx.AttributeProto attr : node.getAttributeList()) {
            if (attr.getName().equals("value") && attr.hasT()) {
                int dataType = attr.getT().getDataType();
                return createTypedOutputType(outputName, dataType, null); // Shape determined at runtime
            }
        }
        // Default to float if no value attribute
        return createTypedOutputType(outputName, Onnx.TensorProto.DataType.FLOAT.getNumber(), null);
    }

    /**
     * Handle Gather operations - preserve data tensor type
     */
    private Onnx.ValueInfoProto handleGatherOperation(Onnx.NodeProto node, String outputName,
                                                      Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        if (!node.getInputList().isEmpty()) {
            String dataInput = node.getInput(0); // First input is data tensor
            if (resolvedTypes.containsKey(dataInput)) {
                return createOutputFromInputType(resolvedTypes.get(dataInput), outputName, node.getOpType());
            }
        }
        return createFallbackOutputType(outputName, node.getOpType());
    }

    /**
     * Handle Expand operation - preserve input type
     */
    private Onnx.ValueInfoProto handleExpandOperation(Onnx.NodeProto node, String outputName,
                                                      Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        if (!node.getInputList().isEmpty()) {
            String inputTensor = node.getInput(0); // First input is the tensor to expand
            if (resolvedTypes.containsKey(inputTensor)) {
                return createOutputFromInputType(resolvedTypes.get(inputTensor), outputName, node.getOpType());
            }
        }
        return createFallbackOutputType(outputName, node.getOpType());
    }

    /**
     * Handle Where operation - output type matches second input (true values)
     */
    private Onnx.ValueInfoProto handleWhereOperation(Onnx.NodeProto node, String outputName,
                                                     Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        if (node.getInputList().size() >= 2) {
            String trueInput = node.getInput(1); // Second input provides the output type
            if (resolvedTypes.containsKey(trueInput)) {
                return createOutputFromInputType(resolvedTypes.get(trueInput), outputName, node.getOpType());
            }
        }
        return createFallbackOutputType(outputName, node.getOpType());
    }

    /**
     * Handle shape manipulation operations - preserve input type
     */
    private Onnx.ValueInfoProto handleShapeManipulationOperation(Onnx.NodeProto node, String outputName,
                                                                 Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        return inheritFromFirstInput(node, outputName, resolvedTypes);
    }

    /**
     * Handle Slice operation - preserve input type
     */
    private Onnx.ValueInfoProto handleSliceOperation(Onnx.NodeProto node, String outputName,
                                                     Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        return inheritFromFirstInput(node, outputName, resolvedTypes);
    }

    /**
     * Handle Concat operation - preserve input type from first input
     */
    private Onnx.ValueInfoProto handleConcatOperation(Onnx.NodeProto node, String outputName,
                                                      Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        return inheritFromFirstInput(node, outputName, resolvedTypes);
    }

    /**
     * Handle Split operation - preserve input type
     */
    private Onnx.ValueInfoProto handleSplitOperation(Onnx.NodeProto node, String outputName,
                                                     Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        return inheritFromFirstInput(node, outputName, resolvedTypes);
    }

    /**
     * Try to inherit type from first input
     */
    private Onnx.ValueInfoProto inheritFromFirstInput(Onnx.NodeProto node, String outputName,
                                                      Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        if (!node.getInputList().isEmpty()) {
            String firstInput = node.getInput(0);
            if (resolvedTypes.containsKey(firstInput)) {
                return createOutputFromInputType(resolvedTypes.get(firstInput), outputName, node.getOpType());
            }
        }
        return null; // Will trigger fallback type creation
    }

    /**
     * Get shape information from first input if available
     */
    private Onnx.TensorShapeProto getShapeFromFirstInput(Onnx.NodeProto node,
                                                         Map<String, Onnx.ValueInfoProto> resolvedTypes) {
        if (!node.getInputList().isEmpty()) {
            String firstInput = node.getInput(0);
            if (resolvedTypes.containsKey(firstInput)) {
                Onnx.ValueInfoProto inputInfo = resolvedTypes.get(firstInput);
                if (inputInfo.getType().hasTensorType() && inputInfo.getType().getTensorType().hasShape()) {
                    return inputInfo.getType().getTensorType().getShape();
                }
            }
        }
        return null;
    }

    /**
     * Create a boolean output type
     */
    private Onnx.ValueInfoProto createBooleanOutputType(String outputName, Onnx.TensorShapeProto shape) {
        return createTypedOutputType(outputName, Onnx.TensorProto.DataType.BOOL.getNumber(), shape);
    }

    /**
     * Create an int64 output type
     */
    private Onnx.ValueInfoProto createInt64OutputType(String outputName, Onnx.TensorShapeProto shape) {
        return createTypedOutputType(outputName, Onnx.TensorProto.DataType.INT64.getNumber(), shape);
    }

    /**
     * Create output type with specific data type and shape
     */
    private Onnx.ValueInfoProto createTypedOutputType(String outputName, int dataType, Onnx.TensorShapeProto shape) {
        Onnx.ValueInfoProto.Builder outputBuilder = Onnx.ValueInfoProto.newBuilder();
        outputBuilder.setName(outputName);

        Onnx.TypeProto.Tensor.Builder tensorBuilder = Onnx.TypeProto.Tensor.newBuilder();
        tensorBuilder.setElemType(dataType);

        if (shape != null) {
            tensorBuilder.setShape(shape);
        }

        Onnx.TypeProto.Builder typeBuilder = Onnx.TypeProto.newBuilder();
        typeBuilder.setTensorType(tensorBuilder.build());
        outputBuilder.setType(typeBuilder.build());

        return outputBuilder.build();
    }

    /**
     * Create output type based on input ValueInfoProto
     */
    private Onnx.ValueInfoProto createOutputFromInputType(Onnx.ValueInfoProto input, String outputName, String opType) {
        Onnx.ValueInfoProto.Builder outputBuilder = Onnx.ValueInfoProto.newBuilder();
        outputBuilder.setName(outputName);

        // Handle operation-specific type changes
        if (opType.equals("Equal") || opType.equals("Greater") || opType.equals("Less") ||
                opType.equals("GreaterOrEqual") || opType.equals("LessOrEqual")) {
            // Comparison operations output BOOL
            Onnx.TypeProto.Tensor.Builder tensorBuilder = Onnx.TypeProto.Tensor.newBuilder();
            tensorBuilder.setElemType(Onnx.TensorProto.DataType.BOOL.getNumber());

            // Preserve shape if available
            if (input.getType().hasTensorType() && input.getType().getTensorType().hasShape()) {
                tensorBuilder.setShape(input.getType().getTensorType().getShape());
            }

            Onnx.TypeProto.Builder typeBuilder = Onnx.TypeProto.newBuilder();
            typeBuilder.setTensorType(tensorBuilder.build());
            outputBuilder.setType(typeBuilder.build());
        } else {
            // Most operations preserve the input type
            outputBuilder.setType(input.getType());
        }

        return outputBuilder.build();
    }

    /**
     * Create a fallback output type when we can't infer the actual type
     * This ensures ALL outputs are available, even if the type might not be perfect
     */
    private Onnx.ValueInfoProto createFallbackOutputType(String outputName, String opType) {
        Onnx.ValueInfoProto.Builder outputBuilder = Onnx.ValueInfoProto.newBuilder();
        outputBuilder.setName(outputName);

        Onnx.TypeProto.Tensor.Builder tensorBuilder = Onnx.TypeProto.Tensor.newBuilder();

        // Use reasonable defaults based on operation type
        switch (opType) {
            case "Equal":
            case "Greater":
            case "Less":
            case "GreaterOrEqual":
            case "LessOrEqual":
            case "And":
            case "Or":
            case "Not":
                tensorBuilder.setElemType(Onnx.TensorProto.DataType.BOOL.getNumber());
                break;
            case "Shape":
                tensorBuilder.setElemType(Onnx.TensorProto.DataType.INT64.getNumber());
                break;
            default:
                // Default to float32 for unknown operations
                tensorBuilder.setElemType(Onnx.TensorProto.DataType.FLOAT.getNumber());
                break;
        }

        Onnx.TypeProto.Builder typeBuilder = Onnx.TypeProto.newBuilder();
        typeBuilder.setTensorType(tensorBuilder.build());
        outputBuilder.setType(typeBuilder.build());

        return outputBuilder.build();
    }

    /**
     * Infer the output type based on node operation and input types
     */
    private Onnx.ValueInfoProto inferOutputType(Onnx.NodeProto node, String outputName,
                                                Map<String, Onnx.ValueInfoProto> valueInfoMap,
                                                Map<String, Onnx.TensorProto> initializerMap,
                                                Map<String, Onnx.ValueInfoProto> inputMap) {
        String opType = node.getOpType();

        // For most operations, output type matches the first input type
        if (!node.getInputList().isEmpty()) {
            String firstInput = node.getInput(0);

            // Check value_info first
            if (valueInfoMap.containsKey(firstInput)) {
                return createOutputFromInputType(valueInfoMap.get(firstInput), outputName, opType);
            }

            // Check initializers
            if (initializerMap.containsKey(firstInput)) {
                return createOutputFromInitializer(initializerMap.get(firstInput), outputName, opType);
            }

            // Check graph inputs
            if (inputMap.containsKey(firstInput)) {
                return createOutputFromInputType(inputMap.get(firstInput), outputName, opType);
            }
        }

        // Handle specific operations that have known output types
        return createOutputFromOperation(outputName, opType, node);
    }



    /**
     * Create output type based on initializer tensor
     */
    private Onnx.ValueInfoProto createOutputFromInitializer(Onnx.TensorProto initializer, String outputName, String opType) {
        Onnx.ValueInfoProto.Builder outputBuilder = Onnx.ValueInfoProto.newBuilder();
        outputBuilder.setName(outputName);

        Onnx.TypeProto.Tensor.Builder tensorBuilder = Onnx.TypeProto.Tensor.newBuilder();

        // Handle operation-specific type changes
        if (opType.equals("Equal") || opType.equals("Greater") || opType.equals("Less")) {
            tensorBuilder.setElemType(Onnx.TensorProto.DataType.BOOL.getNumber());
        } else {
            // Use the initializer's actual data type number
            tensorBuilder.setElemType(initializer.getDataType());
        }

        // Copy shape if available
        if (initializer.getDimsCount() > 0) {
            Onnx.TensorShapeProto.Builder shapeBuilder = Onnx.TensorShapeProto.newBuilder();
            for (long dim : initializer.getDimsList()) {
                Onnx.TensorShapeProto.Dimension.Builder dimBuilder =
                        Onnx.TensorShapeProto.Dimension.newBuilder();
                dimBuilder.setDimValue(dim);
                shapeBuilder.addDim(dimBuilder.build());
            }
            tensorBuilder.setShape(shapeBuilder.build());
        }

        Onnx.TypeProto.Builder typeBuilder = Onnx.TypeProto.newBuilder();
        typeBuilder.setTensorType(tensorBuilder.build());
        outputBuilder.setType(typeBuilder.build());

        return outputBuilder.build();
    }

    /**
     * Create output type based on operation type when input types are unknown
     */
    private Onnx.ValueInfoProto createOutputFromOperation(String outputName, String opType, Onnx.NodeProto node) {
        Onnx.ValueInfoProto.Builder outputBuilder = Onnx.ValueInfoProto.newBuilder();
        outputBuilder.setName(outputName);

        Onnx.TypeProto.Tensor.Builder tensorBuilder = Onnx.TypeProto.Tensor.newBuilder();

        // Determine type based on operation
        switch (opType) {
            case "Equal":
            case "Greater":
            case "Less":
            case "And":
            case "Or":
            case "Not":
                tensorBuilder.setElemType(Onnx.TensorProto.DataType.BOOL.getNumber());
                break;
            case "Shape":
                tensorBuilder.setElemType(Onnx.TensorProto.DataType.INT64.getNumber());
                break;
            case "Cast":
                // Look for 'to' attribute to determine output type
                for (Onnx.AttributeProto attr : node.getAttributeList()) {
                    if (attr.getName().equals("to")) {
                        tensorBuilder.setElemType((int) attr.getI());
                        break;
                    }
                }

                break;
            case "ConstantOfShape":
                // Check value attribute for data type
                for (Onnx.AttributeProto attr : node.getAttributeList()) {
                    if (attr.getName().equals("value") && attr.hasT()) {
                        tensorBuilder.setElemType(attr.getT().getDataType());
                        break;
                    }
                }

                break;
            case "Gather":
                // Gather preserves the data type of the first input (data tensor)
                // but we need to look up the first input to get its type
                if (!node.getInputList().isEmpty()) {
                    // We would need to recursively look up the input type here
                    // For now, return null to indicate we can't infer without more context
                    return null;
                }
                break;
            case "Expand":
                // Expand preserves the input data type
                // but we need the input type - return null for now
                return null;
            case "Where":
                // Where operation: output type matches the second input (true values)
                // but we need to look up the input type
                return null;
            case "Unsqueeze":
                // Unsqueeze preserves input type but changes shape
                return null;
            default:
                // For unknown operations, we can't safely infer the type
                return null;
        }

        Onnx.TypeProto.Builder typeBuilder = Onnx.TypeProto.newBuilder();
        typeBuilder.setTensorType(tensorBuilder.build());
        outputBuilder.setType(typeBuilder.build());

        return outputBuilder.build();
    }

    /**
     * Create or recreate the ONNX Runtime session
     */
    private void createSession() {
        // Close existing session if present
        if (session != null) {
            session.close();
            session = null;
        }

        bp = Loader.getPlatform().toLowerCase().startsWith("windows") ?
                new CharPointer(currentModelPath) : new BytePointer(currentModelPath);
        session = new Session(env, bp, sessionOptions);
        session.retainReference();

        runOptions = new RunOptions();
        memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        sessionNeedsReload = false;
        log.debug("Created new ONNX Runtime session for model: {}", currentModelPath);
    }

    /**
     * Get the default output names from the model
     */
    private List<String> getDefaultOutputNames() {
        List<String> defaultOutputs = new ArrayList<>();
        for (Onnx.ValueInfoProto output : outputs) {
            defaultOutputs.add(output.getName());
        }
        return defaultOutputs;
    }

    /**
     * Validate that all requested output names are available in the model
     */
    private void validateOutputNames(List<String> requestedOutputs) {
        if (requestedOutputs == null || requestedOutputs.isEmpty()) {
            return; // Default outputs are always valid
        }

        for (String outputName : requestedOutputs) {
            if (!allAvailableOutputs.containsKey(outputName)) {
                throw new IllegalArgumentException(
                        String.format("Requested output '%s' is not available in the model. " +
                                "Available outputs: %s", outputName, allAvailableOutputs.keySet()));
            }
        }
    }

    /**
     * Check if outputs are different and reload session if necessary
     */
    private void checkAndReloadSession(List<String> requestedOutputs) {
        if (requestedOutputs == null || requestedOutputs.isEmpty()) {
            requestedOutputs = getDefaultOutputNames();
        }

        validateOutputNames(requestedOutputs);

        // Check if the requested outputs are different from current outputs
        if (!Objects.equals(currentOutputNames, requestedOutputs)) {
            log.info("Outputs are different. Modifying protobuf and reloading EVERYTHING. Previous: {}, New: {}",
                    currentOutputNames, requestedOutputs);

            // Modify protobuf, write to disk, and reload everything
            modifyProtobufAndReload(requestedOutputs);
            currentOutputNames = new ArrayList<>(requestedOutputs);
        }
    }

    /**
     * Modify the protobuf to set custom outputs, write to disk, and reload everything
     */
    private void modifyProtobufAndReload(List<String> customOutputNames) {
        try {
            // Load the original model proto
            Onnx.ModelProto originalModel = Onnx.ModelProto.parseFrom(
                    FileUtils.readFileToByteArray(new File(originalModelUri)));

            // Create a new graph builder from the existing graph
            Onnx.GraphProto.Builder graphBuilder = originalModel.getGraph().toBuilder();

            // Clear existing outputs
            graphBuilder.clearOutput();

            // Add the custom outputs using their ACTUAL types from our cache
            for (String outputName : customOutputNames) {
                Onnx.ValueInfoProto outputInfo = allAvailableOutputs.get(outputName);
                if (outputInfo != null) {
                    graphBuilder.addOutput(outputInfo);
                } else {
                    throw new IllegalArgumentException("Output '" + outputName + "' not found in available outputs");
                }
            }

            // Build the modified model
            Onnx.ModelProto.Builder modelBuilder = originalModel.toBuilder();
            modelBuilder.setGraph(graphBuilder.build());
            Onnx.ModelProto modifiedModel = modelBuilder.build();

            // Write the modified model to a temporary file
            Path tempDir = Files.createTempDirectory("onnx_modified_");
            Path tempModelPath = tempDir.resolve("modified_model.onnx");
            Files.write(tempModelPath, modifiedModel.toByteArray());

            // Update current model path to the modified model
            currentModelPath = tempModelPath.toString();

            // Now reload everything with the modified model
            reloadEverything();

            log.info("Successfully modified protobuf with outputs {} and reloaded model from {}",
                    customOutputNames, currentModelPath);

        } catch (IOException e) {
            log.error("Failed to modify protobuf and reload model", e);
            throw new RuntimeException("Failed to modify model protobuf", e);
        }
    }

    /**
     * Reload everything after protobuf modification
     */
    private void reloadEverything() {
        // Close existing session if present
        if (session != null) {
            session.close();
            session = null;
        }

        // Release all existing resources
        if (sessionOptions != null) {
            sessionOptions.releaseReference();
            sessionOptions = null;
        }
        if (allocator != null) {
            allocator.releaseReference();
            allocator = null;
        }
        if (runOptions != null) {
            runOptions.releaseReference();
            runOptions = null;
        }

        // Clear all model metadata
        initializers.clear();
        inputs.clear();
        outputs.clear();
        if (allAvailableOutputs != null) {
            allAvailableOutputs.clear();
        }

        // Reload everything from the modified model
        loadModel();
        initializeSessionOptions();

        allocator = new OrtAllocator();
        allocator.retainReference();

        createSession();

        sessionNeedsReload = false;
        log.debug("Completely reloaded model, graph, and session for ONNX Runtime: {}", currentModelPath);
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
        }

        if (sessionOptions != null) {
            sessionOptions.releaseReference();
        }
        if (allocator != null) {
            allocator.releaseReference();
        }
        if (runOptions != null) {
            runOptions.releaseReference();
        }

        // Clean up temporary files
        if (!currentModelPath.equals(originalModelUri)) {
            try {
                Path tempModel = Paths.get(currentModelPath);
                if (Files.exists(tempModel)) {
                    Files.delete(tempModel);
                    // Also try to delete the temp directory if it's empty
                    Path tempDir = tempModel.getParent();
                    if (Files.exists(tempDir)) {
                        try {
                            Files.delete(tempDir);
                        } catch (Exception e) {
                            // Ignore if directory is not empty
                        }
                    }
                }
            } catch (IOException e) {
                log.warn("Failed to clean up temporary model file: {}", currentModelPath, e);
            }
        }
    }

    /**
     * Execute the session using the given input Map (backward compatibility)
     * @param input the input map
     * @return a map of the names of the SDValues
     */
    public Map<String, SDValue> execValues(Map<String, SDValue> input) {
        return execValues(input, null);
    }

    /**
     * Execute the session using the given input Map with custom output names
     * @param input the input map
     * @param customOutputNames list of custom output names, null to use default graph outputs
     * @return a map of the names of the SDValues
     */
    public Map<String, SDValue> execValues(Map<String, SDValue> input, List<String> customOutputNames) {
        // Check if outputs are different and reload session if necessary
        checkAndReloadSession(customOutputNames);

        long numInputNodes = session.GetInputCount();
        List<String> outputNames = getOutputNames(customOutputNames);
        long numOutputNodes = outputNames.size();

        PointerPointer<BytePointer> inputNodeNames = new PointerPointer<>(numInputNodes);
        PointerPointer<BytePointer> outputNodeNames = new PointerPointer<>(numOutputNodes);

        Value inputVal = new Value(numInputNodes);
        for (long i = 0; i < numInputNodes; i++) {
            BytePointer inputName = session.GetInputNameAllocated(i, allocator);
            inputNodeNames.put(i, inputName);
            ONNXType typeForInput = getTypeForInput(session, i);
            List<INDArray> arr = input.get(inputName.getString()).getListValue();
            if (arr.size() == 1 && typeForInput == ONNXType.ONNX_TYPE_TENSOR) {
                INDArray arr2 = arr.get(0);
                Value inputTensor = getTensor(arr2, memoryInfo);
                Preconditions.checkState(inputTensor.IsTensor(), "Input must be a tensor.");
                inputVal.position(i).put(inputTensor);
            }
            // empty sequence
            else if (arr.size() == 0) {
                throw new IllegalArgumentException("Onnx Runtime does not support empty sequences! Found at input name " + inputName.getString());
            } else if (arr.size() > 1 || typeForInput == ONNXType.ONNX_TYPE_SEQUENCE) {
                ValueVector inputTensor = getSequence(arr, memoryInfo);
                inputVal.position(i).put(Value.CreateSequence(inputTensor));
            }
        }

        // reset position after iterating
        inputVal.position(0);

        for (int i = 0; i < numOutputNodes; i++) {
            outputNodeNames.put(i, new BytePointer(outputNames.get(i)));
        }

        ValueVector outputVector = session.Run(
                runOptions,
                inputNodeNames,
                inputVal,
                numInputNodes,
                outputNodeNames,
                numOutputNodes);

        outputVector.retainReference();
        Map<String, SDValue> ret = new LinkedHashMap<>();

        for (int i = 0; i < numOutputNodes; i++) {
            Value outValue = outputVector.get(i);
            outValue.retainReference();
            if (outValue.IsTensor()) {
                INDArray arr = getArray(outValue);
                ret.put((outputNodeNames.get(BytePointer.class, i)).getString(), SDValue.create(arr));
            } else {
                INDArray[] seq = ndarraysFromSequence(outValue, allocator);
                ret.put((outputNodeNames.get(BytePointer.class, i)).getString(), SDValue.create(Arrays.asList(seq)));
            }
        }

        return ret;
    }

    /**
     * Execute the session using the given input Map (backward compatibility)
     * @param input the input map
     * @return a map of the names of the ndarrays
     */
    public Map<String, INDArray> exec(Map<String, INDArray> input) {
        return exec(input, null);
    }

    public INDArray getInitializer(String arr) {
        return this.initializers.stream()
                .filter(input -> input.getName().equals(arr))
                .map(input -> OnnxTensorUtils.toINDArray(input))
                .findFirst().orElseThrow();
    }

    /**
     * Execute the session using the given input Map with custom output names
     * @param input the input map
     * @param customOutputNames list of custom output names, null to use default graph outputs
     * @return a map of the names of the ndarrays
     */
    public Map<String, INDArray> exec(Map<String, INDArray> input, List<String> customOutputNames) {
        // Check if outputs are different and reload session if necessary
        checkAndReloadSession(customOutputNames);

        long numInputNodes = session.GetInputCount();
        List<String> outputNames = getOutputNames(customOutputNames);
        long numOutputNodes = outputNames.size();

        PointerPointer<BytePointer> inputNodeNames = new PointerPointer<>(numInputNodes);
        PointerPointer<BytePointer> outputNodeNames = new PointerPointer<>(numOutputNodes);

        Value inputVal = new Value(numInputNodes);

        for (int i = 0; i < numInputNodes; i++) {
            String inputName = inputs.get(i).getName();
            inputNodeNames.put(i, new BytePointer(inputName));
            INDArray arr = input.get(inputName);
            Value inputTensor = getTensor(arr, memoryInfo);
            Preconditions.checkState(inputTensor.IsTensor(), "Input must be a tensor.");
            inputVal.position(i).put(inputTensor);
        }

        // reset position after iterating
        inputVal.position(0);

        for (int i = 0; i < numOutputNodes; i++) {
            outputNodeNames.put(i, new BytePointer(outputNames.get(i)));
        }

        ValueVector outputVector = session.Run(
                runOptions,
                inputNodeNames,
                inputVal,
                numInputNodes,
                outputNodeNames,
                numOutputNodes);

        outputVector.retainReference();
        Map<String, INDArray> ret = new LinkedHashMap<>();

        for (int i = 0; i < numOutputNodes; i++) {
            Value outValue = outputVector.get(i);
            outValue.retainReference();

            // For custom outputs, we need to determine the type dynamically
            ONNXType typeForOutput = getOutputType(outValue, outputNames.get(i));

            switch (typeForOutput) {
                case ONNX_TYPE_SEQUENCE:
                    long count = outValue.GetCount();
                    // Handle sequence outputs if needed
                    INDArray[] seqArrays = ndarraysFromSequence(outValue, allocator);
                    if (seqArrays.length > 0) {
                        // For backward compatibility, return the first array in sequence
                        ret.put(outputNames.get(i), seqArrays[0]);
                    }
                    break;
                case ONNX_TYPE_TENSOR:
                    DataBuffer buffer = getDataBuffer(outValue);
                    LongVector longPointer = outValue.GetTensorTypeAndShapeInfo().GetShape();
                    // shape info can be null
                    if (longPointer != null) {
                        long[] shape = new long[(int)longPointer.size()];
                        for (int j = 0; j < shape.length; j++) {
                            shape[j] = longPointer.get(j);
                        }
                        ret.put(outputNames.get(i), Nd4j.create(buffer).reshape(shape));
                    } else {
                        ret.put(outputNames.get(i), Nd4j.create(buffer));
                    }
                    break;
                case ONNX_TYPE_MAP:
                case ONNX_TYPE_OPAQUE:
                case ONNX_TYPE_UNKNOWN:
                case ONNX_TYPE_OPTIONAL:
                case ONNX_TYPE_SPARSE_TENSOR:
                default:
                    throw new IllegalStateException("Unable to get type " + typeForOutput + " only accepts tensors and sequences.");
            }
        }

        return ret;
    }

    /**
     * Get the list of output names to use for execution
     * @param customOutputNames custom output names, can be null
     * @return list of output names to use
     */
    private List<String> getOutputNames(List<String> customOutputNames) {
        if (customOutputNames != null && !customOutputNames.isEmpty()) {
            return customOutputNames;
        }

        // Default to graph outputs
        return getDefaultOutputNames();
    }

    /**
     * Determine the ONNX type for a given output value and name
     * @param outValue the output value
     * @param outputName the output name
     * @return the ONNXType
     */
    private ONNXType getOutputType(Value outValue, String outputName) {
        if (outValue.IsTensor()) {
            return ONNXType.ONNX_TYPE_TENSOR;
        } else if (outValue.IsSparseTensor()) {
            return ONNXType.ONNX_TYPE_SPARSE_TENSOR;
        } else {
            return ONNXType.ONNX_TYPE_UNKNOWN;
        }
    }

    /**
     * Get all available output names from the model graph (including intermediate nodes)
     * @return set of all available output names
     */
    public Set<String> getAllAvailableOutputNames() {
        return new HashSet<>(allAvailableOutputs.keySet());
    }

    /**
     * Get default model output names
     * @return list of default output names
     */
    public List<String> getAvailableOutputNames() {
        return getDefaultOutputNames();
    }

    /**
     * Get all available input names from the model graph
     * @return list of all input names
     */
    public List<String> getAvailableInputNames() {
        List<String> inputNames = new ArrayList<>();
        for (Onnx.ValueInfoProto input : inputs) {
            inputNames.add(input.getName());
        }
        return inputNames;
    }

    /**
     * Force a session reload on next execution (useful for external configuration changes)
     */
    public void forceSessionReload() {
        this.sessionNeedsReload = true;
        log.info("Forcing complete reload of ONNX session on next execution");
    }

    /**
     * Get the currently configured output names for this session
     * @return list of current output names
     */
    public List<String> getCurrentOutputNames() {
        return new ArrayList<>(currentOutputNames);
    }
}