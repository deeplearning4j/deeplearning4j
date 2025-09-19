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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

    private final String originalModelUri;
    private String processedModelPath; // Path to the processed model with all outputs available

    @Getter
    private List<Onnx.TensorProto> initializers = new ArrayList<>();
    @Getter
    private List<Onnx.ValueInfoProto> inputs = new ArrayList<>();
    @Getter
    private List<Onnx.ValueInfoProto> outputs = new ArrayList<>();

    // Map of ALL possible outputs (including intermediate nodes) with their types
    private Map<String, Onnx.ValueInfoProto> allAvailableOutputs;
    
    // Map of output names that need casting to their cast node outputs
    private Map<String, String> outputCastMapping = new HashMap<>();

    @Builder
    public OnnxRuntimeRunner(String modelUri) {
        this.originalModelUri = modelUri;

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
            // Process the model ONCE to make all outputs available
            processModelForAllOutputs();
            createSession();
        } else {
            runOptions = new RunOptions();
            memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        }
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
     * Process the model to ensure ALL node outputs are available as graph outputs
     * This is done ONCE during initialization
     */
    private void processModelForAllOutputs() {
        try {
            // Load the original model
            modelProto = Onnx.ModelProto.parseFrom(FileUtils.readFileToByteArray(new File(originalModelUri)));
            
            // Extract metadata
            extractModelMetadata();
            
            // Build a complete map of all available outputs with their types
            buildCompleteOutputMap();
            
            // Create a modified model where ALL outputs are available
            Onnx.ModelProto processedModel = createModelWithAllOutputs();
            
            // Save the processed model
            Path tempDir = Files.createTempDirectory("onnx_all_outputs_");
            processedModelPath = tempDir.resolve("all_outputs_model.onnx").toString();
            Files.write(Paths.get(processedModelPath), processedModel.toByteArray());
            
            log.info("Processed model saved to: {} with {} total outputs available", 
                    processedModelPath, allAvailableOutputs.size());
            
        } catch (IOException e) {
            throw new RuntimeException("Failed to process model", e);
        }
    }

    /**
     * Extract model metadata
     */
    private void extractModelMetadata() {
        initializers.clear();
        inputs.clear();
        outputs.clear();

        Onnx.GraphProto graph = modelProto.getGraph();
        
        for (int i = 0; i < graph.getInitializerCount(); i++) {
            initializers.add(graph.getInitializer(i));
        }
        for (int i = 0; i < graph.getInputCount(); i++) {
            inputs.add(graph.getInput(i));
        }
        for (int i = 0; i < graph.getOutputCount(); i++) {
            outputs.add(graph.getOutput(i));
        }
    }

    /**
     * Build a complete map of all available outputs
     */
    private void buildCompleteOutputMap() {
        allAvailableOutputs = new HashMap<>();
        Onnx.GraphProto graph = modelProto.getGraph();
        
        // Add existing graph outputs
        for (Onnx.ValueInfoProto output : graph.getOutputList()) {
            allAvailableOutputs.put(output.getName(), output);
        }
        
        // Add value_info entries
        for (Onnx.ValueInfoProto valueInfo : graph.getValueInfoList()) {
            allAvailableOutputs.put(valueInfo.getName(), valueInfo);
        }
        
        // Add initializers as potential outputs
        for (Onnx.TensorProto initializer : graph.getInitializerList()) {
            if (!allAvailableOutputs.containsKey(initializer.getName())) {
                allAvailableOutputs.put(initializer.getName(), createValueInfoFromInitializer(initializer));
            }
        }
        
        // Add all node outputs
        for (Onnx.NodeProto node : graph.getNodeList()) {
            for (String outputName : node.getOutputList()) {
                if (!allAvailableOutputs.containsKey(outputName)) {
                    // Try to infer type from value_info or create a minimal entry
                    Onnx.ValueInfoProto outputInfo = findOrInferOutputInfo(outputName, graph);
                    allAvailableOutputs.put(outputName, outputInfo);
                }
            }
        }
    }

    /**
     * Find or infer output info for a given output name
     */
    private Onnx.ValueInfoProto findOrInferOutputInfo(String outputName, Onnx.GraphProto graph) {
        // Check value_info first
        for (Onnx.ValueInfoProto valueInfo : graph.getValueInfoList()) {
            if (valueInfo.getName().equals(outputName)) {
                return valueInfo;
            }
        }
        
        // Create minimal info - we'll fix type mismatches later
        return Onnx.ValueInfoProto.newBuilder()
                .setName(outputName)
                .build();
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
     * Create a model where ALL outputs are exposed as graph outputs
     */
    private Onnx.ModelProto createModelWithAllOutputs() {
        Onnx.GraphProto.Builder graphBuilder = modelProto.getGraph().toBuilder();
        
        // Clear existing outputs
        graphBuilder.clearOutput();
        
        // First, try adding all outputs and see what fails
        Map<String, Onnx.ValueInfoProto> outputsToAdd = new HashMap<>(allAvailableOutputs);
        
        // Try to create a test session to detect type mismatches
        boolean hasTypeMismatches = true;
        int maxAttempts = 10; // Prevent infinite loops
        int attempt = 0;
        
        while (hasTypeMismatches && attempt < maxAttempts) {
            attempt++;
            hasTypeMismatches = false;
            
            // Build test model
            graphBuilder.clearOutput();
            for (Onnx.ValueInfoProto output : outputsToAdd.values()) {
                graphBuilder.addOutput(output);
            }
            
            Onnx.ModelProto testModel = modelProto.toBuilder()
                    .setGraph(graphBuilder.build())
                    .build();
            
            // Test for type mismatches
            Map<String, TypeMismatchInfo> mismatches = detectTypeMismatches(testModel);
            
            if (!mismatches.isEmpty()) {
                hasTypeMismatches = true;
                log.info("Attempt {}: Found {} type mismatches, adding cast nodes", attempt, mismatches.size());
                
                // Add cast nodes for each mismatch
                for (Map.Entry<String, TypeMismatchInfo> entry : mismatches.entrySet()) {
                    String outputName = entry.getKey();
                    TypeMismatchInfo mismatch = entry.getValue();
                    
                    // Create a cast node
                    String castOutputName = outputName + "_cast_to_" + mismatch.expectedType.toLowerCase().replace("tensor(", "").replace(")", "");
                    Onnx.NodeProto castNode = createCastNode(outputName, castOutputName, 
                            parseDataType(mismatch.expectedType));
                    graphBuilder.addNode(castNode);
                    
                    // Update the output to use the cast output
                    Onnx.ValueInfoProto castOutputInfo = createOutputInfo(castOutputName, 
                            parseDataType(mismatch.expectedType));
                    outputsToAdd.put(outputName, castOutputInfo);
                    
                    // Store the mapping
                    outputCastMapping.put(outputName, castOutputName);
                }
            }
        }
        
        return modelProto.toBuilder()
                .setGraph(graphBuilder.build())
                .build();
    }

    /**
     * Detect type mismatches by trying to create a test session
     */
    private Map<String, TypeMismatchInfo> detectTypeMismatches(Onnx.ModelProto testModel) {
        Map<String, TypeMismatchInfo> mismatches = new HashMap<>();
        
        try {
            // Save test model temporarily
            Path tempPath = Files.createTempFile("onnx_test_", ".onnx");
            Files.write(tempPath, testModel.toByteArray());
            
            // Try to create a session
            try (Session testSession = new Session(env, new BytePointer(tempPath.toString()), sessionOptions)) {
                // Success - no mismatches
            }
            
            // Clean up
            Files.deleteIfExists(tempPath);
            
        } catch (RuntimeException e) {
            if (e.getMessage() != null && e.getMessage().contains("Type Error")) {
                TypeMismatchInfo mismatch = parseTypeMismatchError(e.getMessage());
                if (mismatch != null) {
                    mismatches.put(mismatch.outputName, mismatch);
                }
            }
        } catch (IOException e) {
            log.error("Failed to test model", e);
        }
        
        return mismatches;
    }

    /**
     * Helper class for type mismatch information
     */
    private static class TypeMismatchInfo {
        String outputName;
        String actualType;
        String expectedType;
        
        TypeMismatchInfo(String outputName, String actualType, String expectedType) {
            this.outputName = outputName;
            this.actualType = actualType;
            this.expectedType = expectedType;
        }
    }

    /**
     * Parse ONNX Runtime type mismatch error to extract information
     */
    private TypeMismatchInfo parseTypeMismatchError(String errorMessage) {
        // Parse error like: "Type (tensor(int32)) of output arg (/encoder/layer.0/attention/self/Reshape_3_output_0) 
        // of node (Attention_0) does not match expected type (tensor(float))."
        
        Pattern pattern = Pattern.compile(
            "Type \\(tensor\\((\\w+)\\)\\) of output arg \\(([^)]+)\\) .* expected type \\(tensor\\((\\w+)\\)\\)"
        );
        
        Matcher matcher = pattern.matcher(errorMessage);
        if (matcher.find()) {
            return new TypeMismatchInfo(
                matcher.group(2),  // output name
                matcher.group(1),  // actual type
                matcher.group(3)   // expected type
            );
        }
        
        return null;
    }

    /**
     * Create a Cast node
     */
    private Onnx.NodeProto createCastNode(String input, String output, int toType) {
        return Onnx.NodeProto.newBuilder()
                .setOpType("Cast")
                .setName(input + "_to_" + getDataTypeName(toType))
                .addInput(input)
                .addOutput(output)
                .addAttribute(Onnx.AttributeProto.newBuilder()
                        .setName("to")
                        .setType(Onnx.AttributeProto.AttributeType.INT)
                        .setI(toType)
                        .build())
                .build();
    }

    /**
     * Create a ValueInfoProto with the specified type
     */
    private Onnx.ValueInfoProto createOutputInfo(String name, int dataType) {
        return Onnx.ValueInfoProto.newBuilder()
                .setName(name)
                .setType(Onnx.TypeProto.newBuilder()
                        .setTensorType(Onnx.TypeProto.Tensor.newBuilder()
                                .setElemType(dataType)
                                .build())
                        .build())
                .build();
    }

    /**
     * Parse data type from string representation
     */
    private int parseDataType(String typeStr) {
        // Parse strings like "tensor(float)" or "float" or "float32"
        typeStr = typeStr.toLowerCase()
                .replace("tensor(", "")
                .replace(")", "")
                .trim();
        
        switch (typeStr) {
            case "float":
            case "float32":
                return Onnx.TensorProto.DataType.FLOAT.getNumber();
            case "double":
            case "float64":
                return Onnx.TensorProto.DataType.DOUBLE.getNumber();
            case "int32":
            case "int":
                return Onnx.TensorProto.DataType.INT32.getNumber();
            case "int64":
            case "long":
                return Onnx.TensorProto.DataType.INT64.getNumber();
            case "bool":
            case "boolean":
                return Onnx.TensorProto.DataType.BOOL.getNumber();
            case "int8":
                return Onnx.TensorProto.DataType.INT8.getNumber();
            case "int16":
                return Onnx.TensorProto.DataType.INT16.getNumber();
            case "uint8":
                return Onnx.TensorProto.DataType.UINT8.getNumber();
            case "uint16":
                return Onnx.TensorProto.DataType.UINT16.getNumber();
            case "uint32":
                return Onnx.TensorProto.DataType.UINT32.getNumber();
            case "uint64":
                return Onnx.TensorProto.DataType.UINT64.getNumber();
            case "float16":
            case "half":
                return Onnx.TensorProto.DataType.FLOAT16.getNumber();
            default:
                log.warn("Unknown type string: {}, defaulting to FLOAT", typeStr);
                return Onnx.TensorProto.DataType.FLOAT.getNumber();
        }
    }

    /**
     * Get human-readable name for ONNX data type
     */
    private String getDataTypeName(int dataType) {
        switch (dataType) {
            case 1: return "FLOAT";
            case 2: return "UINT8";
            case 3: return "INT8";
            case 4: return "UINT16";
            case 5: return "INT16";
            case 6: return "INT32";
            case 7: return "INT64";
            case 8: return "STRING";
            case 9: return "BOOL";
            case 10: return "FLOAT16";
            case 11: return "DOUBLE";
            case 12: return "UINT32";
            case 13: return "UINT64";
            case 14: return "COMPLEX64";
            case 15: return "COMPLEX128";
            case 16: return "BFLOAT16";
            default: return "UNKNOWN(" + dataType + ")";
        }
    }

    /**
     * Create the ONNX Runtime session with the processed model
     */
    private void createSession() {
        bp = Loader.getPlatform().toLowerCase().startsWith("windows") ?
                new CharPointer(processedModelPath) : new BytePointer(processedModelPath);
        session = new Session(env, bp, sessionOptions);
        session.retainReference();

        runOptions = new RunOptions();
        memoryInfo = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        log.debug("Created ONNX Runtime session with all outputs available");
    }

    public Map<String,INDArray> getConstantsOrInitializers(List<String> names) {
        Map<String,INDArray> ret = new LinkedHashMap<>();
        for(String name : names) {
            ret.put(name,getConstantOrInitializer(name));
        }
        return ret;
    }

    public INDArray getConstantOrInitializer(String name) {
        return this.initializers.stream().filter(input -> input.getName().equals(name))
                .map(input -> OnnxTensorUtils.toINDArray(input)).findAny().orElseThrow();
    }

    public INDArray getInitializer(String arr) {
        return this.initializers.stream()
                .filter(input -> input.getName().equals(arr))
                .map(input -> OnnxTensorUtils.toINDArray(input))
                .findFirst().orElseThrow();
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
        // Validate inputs first
        if (input == null || input.isEmpty()) {
            throw new IllegalArgumentException("No inputs provided. Model expects the following inputs: " + 
                    getAvailableInputNames());
        }
        
        if (customOutputNames == null || customOutputNames.isEmpty()) {
            customOutputNames = getDefaultOutputNames();
        }
        
        // Validate outputs exist
        for (String output : customOutputNames) {
            if (!allAvailableOutputs.containsKey(output) && !outputCastMapping.containsKey(output)) {
                throw new IllegalArgumentException(
                    String.format("Output '%s' not available. Available outputs: %s", 
                        output, allAvailableOutputs.keySet()));
            }
        }
        
        // Map requested outputs to actual outputs (handling casts)
        List<String> actualOutputs = new ArrayList<>();
        Map<String, String> reverseMapping = new HashMap<>();
        
        for (String output : customOutputNames) {
            String actualOutput = outputCastMapping.getOrDefault(output, output);
            actualOutputs.add(actualOutput);
            reverseMapping.put(actualOutput, output);
        }

        long numInputNodes = session.GetInputCount();
        long numOutputNodes = actualOutputs.size();

        PointerPointer<BytePointer> inputNodeNames = new PointerPointer<>(numInputNodes);
        PointerPointer<BytePointer> outputNodeNames = new PointerPointer<>(numOutputNodes);

        Value inputVal = new Value(numInputNodes);
        for (long i = 0; i < numInputNodes; i++) {
            BytePointer inputName = session.GetInputNameAllocated(i, allocator);
            inputNodeNames.put(i, inputName);
            String inputNameStr = inputName.getString();
            
            if (!input.containsKey(inputNameStr)) {
                throw new IllegalArgumentException("Missing required input: '" + inputNameStr + 
                        "'. Provided inputs: " + input.keySet());
            }
            
            ONNXType typeForInput = getTypeForInput(session, i);
            List<INDArray> arr = input.get(inputNameStr).getListValue();
            if (arr.size() == 1 && typeForInput == ONNXType.ONNX_TYPE_TENSOR) {
                INDArray arr2 = arr.get(0);
                Value inputTensor = getTensor(arr2, memoryInfo);
                Preconditions.checkState(inputTensor.IsTensor(), "Input must be a tensor.");
                inputVal.position(i).put(inputTensor);
            }
            // empty sequence
            else if (arr.size() == 0) {
                throw new IllegalArgumentException("Onnx Runtime does not support empty sequences! Found at input name " + inputNameStr);
            } else if (arr.size() > 1 || typeForInput == ONNXType.ONNX_TYPE_SEQUENCE) {
                ValueVector inputTensor = getSequence(arr, memoryInfo);
                inputVal.position(i).put(Value.CreateSequence(inputTensor));
            }
        }

        // reset position after iterating
        inputVal.position(0);

        for (int i = 0; i < numOutputNodes; i++) {
            outputNodeNames.put(i, new BytePointer(actualOutputs.get(i)));
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
            
            String actualOutputName = actualOutputs.get(i);
            String requestedOutputName = reverseMapping.getOrDefault(actualOutputName, actualOutputName);
            
            if (outValue.IsTensor()) {
                INDArray arr = getArray(outValue);
                ret.put(requestedOutputName, SDValue.create(arr));
            } else {
                INDArray[] seq = ndarraysFromSequence(outValue, allocator);
                ret.put(requestedOutputName, SDValue.create(Arrays.asList(seq)));
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

    /**
     * Execute the session using the given input Map with custom output names
     * @param input the input map
     * @param customOutputNames list of custom output names, null to use default graph outputs
     * @return a map of the names of the ndarrays
     */
    public Map<String, INDArray> exec(Map<String, INDArray> input, List<String> customOutputNames) {
        // Validate inputs first
        if (input == null || input.isEmpty()) {
            throw new IllegalArgumentException("No inputs provided. Model expects the following inputs: " + 
                    getAvailableInputNames());
        }
        
        // Check that all required inputs are present
        for (Onnx.ValueInfoProto inputInfo : inputs) {
            String inputName = inputInfo.getName();
            if (!input.containsKey(inputName)) {
                throw new IllegalArgumentException("Missing required input: '" + inputName + 
                        "'. Provided inputs: " + input.keySet());
            }
            
            // Validate non-null
            INDArray arr = input.get(inputName);
            if (arr == null) {
                throw new IllegalArgumentException("Input '" + inputName + "' is null");
            }
        }
        
        if (customOutputNames == null || customOutputNames.isEmpty()) {
            customOutputNames = getDefaultOutputNames();
        }
        
        // Validate outputs exist
        for (String output : customOutputNames) {
            if (!allAvailableOutputs.containsKey(output) && !outputCastMapping.containsKey(output)) {
                throw new IllegalArgumentException(
                    String.format("Output '%s' not available. Available outputs: %s", 
                        output, allAvailableOutputs.keySet()));
            }
        }
        
        // Map requested outputs to actual outputs (handling casts)
        List<String> actualOutputs = new ArrayList<>();
        Map<String, String> reverseMapping = new HashMap<>();
        
        for (String output : customOutputNames) {
            String actualOutput = outputCastMapping.getOrDefault(output, output);
            actualOutputs.add(actualOutput);
            reverseMapping.put(actualOutput, output);
        }

        long numInputNodes = session.GetInputCount();
        long numOutputNodes = actualOutputs.size();

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
            outputNodeNames.put(i, new BytePointer(actualOutputs.get(i)));
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
            
            String actualOutputName = actualOutputs.get(i);
            String requestedOutputName = reverseMapping.getOrDefault(actualOutputName, actualOutputName);

            DataBuffer buffer = getDataBuffer(outValue);
            LongVector longPointer = outValue.GetTensorTypeAndShapeInfo().GetShape();
            // shape info can be null
            if (longPointer != null) {
                long[] shape = new long[(int)longPointer.size()];
                for (int j = 0; j < shape.length; j++) {
                    shape[j] = longPointer.get(j);
                }
                ret.put(requestedOutputName, Nd4j.create(buffer).reshape(shape));
            } else {
                ret.put(requestedOutputName, Nd4j.create(buffer));
            }
        }

        return ret;
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
     * Save temporary model to disk
     */
    private Path saveTemporaryModel(Onnx.ModelProto model) throws IOException {
        Path tempDir = Files.createTempDirectory("onnx_test_");
        Path tempModelPath = tempDir.resolve("test_model.onnx");
        Files.write(tempModelPath, model.toByteArray());
        return tempModelPath;
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
        if (processedModelPath != null) {
            try {
                Path path = Paths.get(processedModelPath);
                Files.deleteIfExists(path);
                Files.deleteIfExists(path.getParent());
            } catch (IOException e) {
                log.warn("Failed to clean up temporary files", e);
            }
        }
    }
}
