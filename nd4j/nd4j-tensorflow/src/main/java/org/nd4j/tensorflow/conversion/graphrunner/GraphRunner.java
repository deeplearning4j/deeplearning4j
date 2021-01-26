/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.tensorflow.conversion.graphrunner;

import lombok.*;
import org.apache.commons.io.FileUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.shade.protobuf.ByteString;
import org.nd4j.shade.protobuf.InvalidProtocolBufferException;
import org.nd4j.shade.protobuf.util.JsonFormat;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.tensorflow.conversion.TensorDataType;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.TensorflowConversion;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.NodeDef;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import org.bytedeco.tensorflow.*;
import static org.bytedeco.tensorflow.global.tensorflow.*;

/**
 * Runs a tensorflow session based on zero copy
 * {@link INDArray} memory replicated to tensorflow.
 *
 * {@link INDArray} is used to hold the memory
 * while tensorflow's c bindings are  used for running the graph.
 *
 * @author Adam Gibson
 */
@Slf4j
@NoArgsConstructor
public class GraphRunner implements Closeable {

    private static boolean isTfWarmedUp = false;
    private static boolean isTfWarmingUp = false;
    private SavedModelConfig savedModelConfig;
    //the in memory representation parsed from protobuf
    private TF_Graph graph;
    //the conversion between nd4j and TensorFlow
    private TensorflowConversion conversion =  TensorflowConversion.getInstance();
    //a persistent session to be used when running the graph
    private TF_Session session;
    //the options for the model
    private TF_SessionOptions options;
    //a status object used
    private TF_Status status;
    @Getter
    @Setter
    @Singular
    private List<String> inputOrder,outputOrder;
    @Getter
    private org.tensorflow.framework.ConfigProto sessionOptionsConfigProto;
    @Getter
    @Setter
    @Singular
    private Map<String,TensorDataType> inputDataTypes,outputDataTypes;
    private static Map<Pair<TensorDataType,TensorDataType>,GraphRunner> recastGraphDefs;

    static {
        recastGraphDefs = new ConcurrentHashMap<>();
    }


    /**
     * The constructor for creating a graph runner via builder
     * @param inputNames the input names to use
     * @param outputNames the output names to use
     * @param savedModelConfig the saved model configuration to load from (note this can not be used in conjunction
     *                         with graph path)
     * @param sessionOptionsConfigProto the session options for running the model (this maybe null)
     * @param sessionOptionsProtoBytes the proto bytes equivalent of the session configuration
     * @param sessionOptionsProtoPath the file path to a session configuration proto file
     * @param graph the tensorflow graph to use
     * @param graphPath the path to the graph
     * @param graphBytes the in memory bytes of the graph
     * @param inputDataTypes the expected input data types
     * @param outputDataTypes the expected output data types
     */



    @Builder
    public GraphRunner(List<String> inputNames,
                       List<String> outputNames,
                       SavedModelConfig savedModelConfig,
                       org.tensorflow.framework.ConfigProto sessionOptionsConfigProto,
                       byte[] sessionOptionsProtoBytes,
                       File sessionOptionsProtoPath,
                       TF_Graph graph,
                       File graphPath,
                       byte[] graphBytes,
                       Map<String, TensorDataType> inputDataTypes,
                       Map<String, TensorDataType> outputDataTypes) {
        try {
            if(sessionOptionsConfigProto == null) {
                if(sessionOptionsConfigProto != null) {
                    this.sessionOptionsConfigProto = ConfigProto.parseFrom(sessionOptionsProtoBytes);
                }
                else if(sessionOptionsProtoPath != null) {
                    byte[] load = FileUtils.readFileToByteArray(sessionOptionsProtoPath);
                    this.sessionOptionsConfigProto = ConfigProto.parseFrom(load);
                }
            }
            else
                this.sessionOptionsConfigProto = sessionOptionsConfigProto;


            this.inputDataTypes = inputDataTypes;
            this.outputDataTypes = outputDataTypes;
            //note that the input and output order, maybe null here
            //if the names are specified, we should defer to those instead
            this.inputOrder = inputNames;
            this.outputOrder = outputNames;
            initOptionsIfNeeded();

            if(graph != null) {
                this.graph = graph;
            }
            else if(graphBytes != null) {
                this.graph = conversion.loadGraph(graphBytes, status);
            }
            else if(graphPath != null) {
                graphBytes = IOUtils.toByteArray(graphPath.toURI());
                this.graph = conversion.loadGraph(graphBytes, status);
            }
            else
                this.graph = TF_NewGraph();

            if(savedModelConfig != null) {
                this.savedModelConfig = savedModelConfig;
                Map<String,String> inputsMap = new LinkedHashMap<>();
                Map<String,String> outputsMap = new LinkedHashMap<>();

                this.session = conversion.loadSavedModel(savedModelConfig, options, null, this.graph, inputsMap, outputsMap, status);

                if(inputOrder == null || inputOrder.isEmpty())
                    inputOrder = new ArrayList<>(inputsMap.values());
                if(outputOrder == null || outputOrder.isEmpty())
                    outputOrder = new ArrayList<>(outputsMap.values());

                savedModelConfig.setSavedModelInputOrder(new ArrayList<>(inputsMap.values()));
                savedModelConfig.setSaveModelOutputOrder(new ArrayList<>(outputsMap.values()));
                log.info("Loaded input names from saved model configuration " + inputOrder);
                log.info("Loaded output names from saved model configuration " + outputOrder);

            }


            initSessionAndStatusIfNeeded(graphBytes);
        } catch (Exception e) {
            throw new IllegalArgumentException("Unable to parse protobuf",e);
        }
    }



    /**
     * Cast inputs from the original data type
     * to the target resulting input data type.
     * This is for when there's a disconnect from the inputs
     * to the target input data type. This runs a pre cast automatically.
     * @param inputs the inputs to cast
     * @return the re casted input
     */
    public Map<String, TF_Tensor> recastInputs(Map<String, TF_Tensor> inputs) {
        return recastInputs(inputs,inputOrder,inputDataTypes);
    }


    /**
     * Cast inputs from the original data type
     * to the target resulting input data type.
     * This is for when there's a disconnect from the inputs
     * to the target input data type. This runs a pre cast automatically.
     * @param inputs the inputs to cast
     * @return the re casted input
     */
    public Map<String, TF_Tensor> recastOutputs(Map<String, TF_Tensor> inputs) {
        return recastInputs(inputs,outputOrder,outputDataTypes);
    }


    /**
     * Automatically recast the input arrays
     * as the specified types
     * @param inputs the input tensors to recast
     * @param inputOrder the order of the input tensors
     * @param inputDataTypes the data types to cast to (null means stay the same)
     * @return the new values
     */
    public Map<String, TF_Tensor> recastInputs(Map<String, TF_Tensor> inputs, List<String> inputOrder, Map<String,TensorDataType> inputDataTypes) {
        if(inputDataTypes == null || inputDataTypes.isEmpty()) {

            inputDataTypes = new LinkedHashMap<>();
            for(int i = 0; i < inputOrder.size(); i++) {
                TensorDataType tensorDataType = TensorDataType.values()[TF_TensorType(inputs.get(inputOrder.get(i)))];
                Preconditions.checkNotNull(tensorDataType,"Data type of " + TF_TensorType(inputs.get(inputOrder.get(i))) + " was null!");
                inputDataTypes.put(inputOrder.get(i),tensorDataType);
            }
        }

        Map<String, TF_Tensor> ret = new HashMap<>();
        for(int i = 0; i < inputOrder.size(); i++) {
            TF_Tensor currInput = inputs.get(inputOrder.get(i));
            TensorDataType fromDType = TensorDataType.values()[TF_TensorType(currInput)];
            if(fromDType != inputDataTypes.get(inputOrder.get(i))) {
                TF_Tensor oldTensor = currInput;
                currInput = castTensor(currInput, fromDType, inputDataTypes.get(inputOrder.get(i)));
                TF_DeleteTensor(oldTensor);
            }

            ret.put(inputOrder.get(i),currInput);
        }

        return ret;
    }

    /**
     * Run the graph definition with the given inputs
     * in native tensorflow
     * @param inputs the inputs to run
     * @return the outputSchema from the native tensorflow wrapper
     */
    public Map<String, TF_Tensor> runTfTensor(Map<String, TF_Tensor> inputs) {
        if(graph == null) {
            throw new IllegalStateException("Graph not initialized.");
        }


        if(inputs.size() != inputOrder.size()) {
            throw new IllegalArgumentException("Number of inputs specified do not match number of arrays specified.");
        }

        if(inputDataTypes == null) {
            inputDataTypes = new LinkedHashMap<>();
            for(int i = 0; i < inputOrder.size(); i++) {
                inputDataTypes.put(inputOrder.get(i),TensorDataType.values()[TF_TensorType(inputs.get(inputOrder.get(i)))]);
            }
        }

        for(Map.Entry<String, org.bytedeco.tensorflow.TF_Tensor> entry : inputs.entrySet()) {
            Preconditions.checkNotNull(entry.getValue(),"Entry " + entry.getKey() + " was null!");
        }

        //recast for adapting input
        inputs = recastInputs(inputs);


        if(savedModelConfig != null) {
            Map<String, TF_Tensor> outputArrays = new LinkedHashMap<>();

            Map<String, org.bytedeco.tensorflow.TF_Operation> opsByName = new HashMap<>();
            org.bytedeco.tensorflow.TF_Output inputOut = new org.bytedeco.tensorflow.TF_Output(savedModelConfig.getSavedModelInputOrder().size());

            TF_Tensor[] inputTensors = new TF_Tensor[savedModelConfig.getSavedModelInputOrder().size()];
            for(int i = 0; i < savedModelConfig.getSavedModelInputOrder().size(); i++) {
                String[] name = savedModelConfig.getSavedModelInputOrder().get(i).split(":");
                org.bytedeco.tensorflow.TF_Operation inputOp = TF_GraphOperationByName(graph, name[0]);
                opsByName.put(savedModelConfig.getSavedModelInputOrder().get(i),inputOp);
                inputOut.position(i).oper(inputOp).index(name.length > 1 ? Integer.parseInt(name[1]) : 0);
                TF_Tensor tfTensor = inputs.get(inputOrder != null && !inputOrder.isEmpty()
                        ? inputOrder.get(i) : savedModelConfig.getSavedModelInputOrder().get(i));
                inputTensors[i] = tfTensor;
            }


            //reset the position of the pointer for execution
            inputOut.position(0);

            org.bytedeco.tensorflow.TF_Output outputOut = new org.bytedeco.tensorflow.TF_Output(savedModelConfig.getSaveModelOutputOrder().size());
            //only setup the output ops
            for(int i = 0; i < savedModelConfig.getSaveModelOutputOrder().size(); i++) {
                String[] name = savedModelConfig.getSaveModelOutputOrder().get(i).split(":");
                org.bytedeco.tensorflow.TF_Operation outputOp = TF_GraphOperationByName(graph, name[0]);
                opsByName.put(savedModelConfig.getSaveModelOutputOrder().get(i),outputOp);
                outputOut.position(i).oper(outputOp).index(name.length > 1 ? Integer.parseInt(name[1]) : 0);
            }

            //reset the position of the pointer for execution
            outputOut.position(0);



            //these are references to the nd4j ndarrays wrapped for tensorflow
            PointerPointer<TF_Tensor> inputTensorsPointer = new PointerPointer<>(inputTensors);
            //note that these are the result pointers
            //the result pointers are null, and will be populated automatically by the session run
            PointerPointer<TF_Tensor> outputTensorsPointer = new PointerPointer<>(savedModelConfig.getSaveModelOutputOrder().size());

            long start = System.nanoTime();
            TF_SessionRun(
                    session,
                    null,
                    //inputs
                    inputOut, inputTensorsPointer, inputTensors.length,
                    //outputSchema
                    outputOut, outputTensorsPointer, savedModelConfig.getSaveModelOutputOrder().size(),
                    //targets
                    null, 0,
                    null,
                    status);        long end = System.nanoTime();
            long diff = TimeUnit.NANOSECONDS.toMillis((end - start));
            log.debug("Session runtime: {} ms", diff);




            if (TF_GetCode(status) != TF_OK) {
                throw new IllegalStateException("ERROR: Unable to run session " + TF_Message(status).getString());
            } else {
                for(int i = 0; i < outputOrder.size(); i++) {
                    outputArrays.put(outputOrder != null && !outputOrder.isEmpty() ? outputOrder.get(i) :
                            savedModelConfig.getSaveModelOutputOrder().get(i),new TF_Tensor(outputTensorsPointer.get(i)));
                }

            }

            return outputArrays;

        }
        else {
            Map<String, TF_Tensor> outputArrays = new LinkedHashMap<>();

            Map<String, org.bytedeco.tensorflow.TF_Operation> opsByName = new HashMap<>();
            org.bytedeco.tensorflow.TF_Output inputOut = new org.bytedeco.tensorflow.TF_Output(inputOrder.size());

            TF_Tensor[] inputTensors = new TF_Tensor[inputOrder.size()];
            for(int i = 0; i < inputOrder.size(); i++) {
                String[] name = inputOrder.get(i).split(":");
                org.bytedeco.tensorflow.TF_Operation inputOp = TF_GraphOperationByName(graph, name[0]);
                opsByName.put(inputOrder.get(i),inputOp);
                inputOut.position(i).oper(inputOp).index(name.length > 1 ? Integer.parseInt(name[1]) : 0);
                TF_Tensor tf_tensor = inputs.get(inputOrder.get(i));

                inputTensors[i] = tf_tensor;
            }


            //reset the position of the pointer for execution
            inputOut.position(0);

            org.bytedeco.tensorflow.TF_Output outputOut = new org.bytedeco.tensorflow.TF_Output(outputOrder.size());
            //only setup the output ops
            for(int i = 0; i < outputOrder.size(); i++) {
                String[] name = outputOrder.get(i).split(":");
                org.bytedeco.tensorflow.TF_Operation outputOp = TF_GraphOperationByName(graph, name[0]);
                if(outputOp == null) {
                    throw new IllegalArgumentException("Illegal output found " + outputOrder.get(i) + " - no op found! Mis specified name perhaps?");
                }

                opsByName.put(outputOrder.get(i),outputOp);
                outputOut.position(i).oper(outputOp).index(name.length > 1 ? Integer.parseInt(name[1]) : 0);
            }

            //reset the position of the pointer for execution
            outputOut.position(0);



            //these are references to the nd4j ndarrays wrapped for tensorflow
            PointerPointer<TF_Tensor> inputTensorsPointer = new PointerPointer<>(inputTensors);
            //note that these are the result pointers
            //the result pointers are null, and will be populated automatically by the session run
            PointerPointer<TF_Tensor> outputTensorsPointer = new PointerPointer<>(outputOrder.size());

            long start = System.nanoTime();
            TF_SessionRun(
                    session,
                    null,
                    //inputs
                    inputOut, inputTensorsPointer, inputOrder.size(),
                    //output
                    outputOut, outputTensorsPointer, outputOrder.size(),
                    //targets
                    null, 0,
                    null,
                    status);
            long end = System.nanoTime();
            long diff = TimeUnit.NANOSECONDS.toMillis((end - start));
            log.debug("Session runtime: {} ms", diff);






            if (TF_GetCode(status) != TF_OK) {
                throw new IllegalStateException("ERROR: Unable to run session " + TF_Message(status).getString());
            } else {
                for(int i = 0; i < outputOrder.size(); i++) {
                    outputArrays.put(outputOrder.get(i),new TF_Tensor(outputTensorsPointer.get(i)));
                }
            }

            return outputArrays;
        }
    }


    /**
     * Returns a map of the output names
     * to the ndarrays matching each output.
     *
     * Note that {@link IllegalArgumentException}
     * will be thrown if there are any invalid states such as:
     * the graph being null
     *
     *
     * the inputs resolved from the graph do not match
     * the inputs passed in
     *
     *
     *
     * @param inputs the inputs to use for each
     *               {@link INDArray}
     * @return a map of the output names to the
     * ndarrays matching each output specified in the graph
     */

    public Map<String,INDArray> run(Map<String,INDArray> inputs) {
        if (!isTfWarmedUp && !isTfWarmingUp){
            isTfWarmingUp = true;
            run(inputs);
            isTfWarmedUp = true;
        }
        Map<String, TF_Tensor> inputTensors = new LinkedHashMap<>();
        for(Map.Entry<String,INDArray> input : inputs.entrySet()) {
            inputTensors.put(input.getKey(),conversion.tensorFromNDArray(input.getValue()));
        }

        Map<String, TF_Tensor> outputTensors = runTfTensor(inputTensors);
        Map<String,INDArray> output = new LinkedHashMap<>();
        for(Map.Entry<String, TF_Tensor> outputTensor : outputTensors.entrySet()) {
            output.put(outputTensor.getKey(),conversion.ndArrayFromTensor(outputTensor.getValue()));
        }

        return output;
    }


    private void initOptionsIfNeeded() {
        //setup the status object to be used for all tensorflow calls
        if(status == null) {
            status = TF_NewStatus();
        }

        if (options == null) {
            options = TF_NewSessionOptions();
            if(sessionOptionsConfigProto != null) {
                BytePointer bytePointer = new BytePointer(sessionOptionsConfigProto.toByteArray());
                TF_SetConfig(options,bytePointer,bytePointer.getStringBytes().length,status);
                if (TF_GetCode(status) != TF_OK) {
                    throw new IllegalStateException("ERROR: Unable to set value configuration:" + TF_Message(status).getString());
                }
            }
        }
    }

    private void initSessionAndStatusIfNeeded(org.tensorflow.framework.GraphDef graphDef1) {
        //infer the inputs and outputSchema for the graph
        Set<String> seenAsInput = new LinkedHashSet<>();
        for(int i = 0; i < graphDef1.getNodeCount(); i++) {
            NodeDef node = graphDef1.getNode(i);
            for(int input = 0; input < node.getInputCount(); input++) {
                seenAsInput.add(node.getInput(input));
            }
        }

        if(outputOrder == null) {
            outputOrder = new ArrayList<>();
            log.trace("Attempting to automatically resolve tensorflow output names..");
            //find the nodes that were not inputs to any  nodes: these are the outputSchema
            for(int i = 0; i < graphDef1.getNodeCount(); i++) {
                if(!seenAsInput.contains(graphDef1.getNode(i).getName()) && !graphDef1.getNode(i).getOp().equals("Placeholder")) {
                    outputOrder.add(graphDef1.getNode(i).getName());
                }
            }

            //multiple names: purge any generated names from the output
            if(outputOrder.size() > 1) {
                Set<String> remove = new HashSet<>();
                for (String name : outputOrder) {
                    if(name.contains("/")) {
                        remove.add(name);
                    }
                }

                outputOrder.removeAll(remove);
            }
        }


        //setup and configure the session, factoring
        //in the ConfigObject as needed
        if(session == null) {
            initOptionsIfNeeded();
            session = TF_NewSession(graph, options, status);
            if (TF_GetCode(status) != TF_OK) {
                throw new IllegalStateException("ERROR: Unable to open session " + TF_Message(status).getString());
            }

        }

    }

    private void initSessionAndStatusIfNeeded(byte[] graphToUse) {
        if(graphToUse == null) {
            //saved model configuration
            return;
        }

        try {
            //use the protobuf api to load the graph definition and load the node metadata
            org.tensorflow.framework.GraphDef graphDef1 = org.tensorflow.framework.GraphDef.parseFrom(graphToUse);
            initSessionAndStatusIfNeeded(graphDef1);
        } catch (org.nd4j.shade.protobuf.InvalidProtocolBufferException e) {
            log.error("",e);
        }
    }


    /**
     * Convert a json string written out
     * by {@link org.nd4j.shade.protobuf.util.JsonFormat}
     * to a {@link org.bytedeco.tensorflow.ConfigProto}
     * @param json the json to read
     * @return the config proto to use
     */
    public static org.tensorflow.framework.ConfigProto fromJson(String json) {
        org.tensorflow.framework.ConfigProto.Builder builder = org.tensorflow.framework.ConfigProto.newBuilder();
        try {
            org.nd4j.shade.protobuf.util.JsonFormat.parser().merge(json,builder);
            org.tensorflow.framework.ConfigProto build = builder.build();
            org.nd4j.shade.protobuf.ByteString serialized = build.toByteString();
            byte[] binaryString = serialized.toByteArray();
            org.tensorflow.framework.ConfigProto configProto = org.tensorflow.framework.ConfigProto.parseFrom(binaryString);
            return configProto;
        } catch (Exception e) {
            log.error("",e);
        }

        return null;
    }


    /**
     * Cast a tensor to another type using
     * the tensorflow c api.
     * This method loads a graph from the classpath from
     * cast_graph/cast_(name of datatype lower case).pb
     * which contains a simple protobuf file with a
     * variant data type tensorflow input place holder
     * named place holder and an output named cast_output.
     * @param input  the input data
     * @param from the input data type to cast from
     * @param to the output data type to
     * @return the casted tensor
     */
    public static TF_Tensor castTensor(TF_Tensor input, TensorDataType from, TensorDataType to) {
        if(from.equals(to))
            return input;

        Map<String, TF_Tensor> inputMap = new HashMap<>();
        inputMap.put("input",input);
        GraphRunner graphRunner = getRunner(from,to);
        try {
            Map<String, TF_Tensor> output = graphRunner.runTfTensor(inputMap);
            return output.get("cast_output");

        } catch(Exception e) {
            throw new IllegalStateException("Unable to run graph",e);
        }
    }

    private static GraphRunner getRunner(TensorDataType from,TensorDataType to) {
        Pair<TensorDataType,TensorDataType> key = Pair.of(from,to);
        if(!recastGraphDefs.containsKey(key)) {
            byte[] graphForDataType = graphForDataType(from,to);
            GraphRunner graphRunner = GraphRunner.builder()
                    .graphBytes(graphForDataType)
                    .inputNames(Arrays.asList("input"))
                    .outputNames(Arrays.asList("cast_output"))
                    .build();

            recastGraphDefs.put(key,graphRunner);
            return graphRunner;
        }

        return recastGraphDefs.get(key);
    }


    private static byte[] graphForDataType(TensorDataType from,TensorDataType to) {
        ClassPathResource classPathResource = new ClassPathResource("cast_graph/cast_" + TensorDataType.toPythonName(from) +  "_" + TensorDataType.toPythonName(to) + ".pb");
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        try (InputStream is = classPathResource.getInputStream()) {
            IOUtils.copy(is, byteArrayOutputStream);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read graph " + classPathResource.getFilename(),e);
        }

        return byteArrayOutputStream.toByteArray();
    }

    /**
     * Write out the session options used
     * by this {@link org.nd4j.tensorflow.conversion.graphrunner.GraphRunner}
     * a s a  json string using the
     * {@link org.nd4j.shade.protobuf.util.JsonFormat}
     * @return the session options as json (mainly for debugging)
     */
    public String sessionOptionsToJson() {
        if(sessionOptionsConfigProto == null)
            return null;
        try {
            return org.nd4j.shade.protobuf.util.JsonFormat.printer().print(sessionOptionsConfigProto);
        } catch (Exception e) {
            log.error("",e);
        }

        return null;
    }


    @Override
    public void close() {
        if(session != null && status != null) {
            TF_CloseSession(session, status);
            TF_DeleteSession(session,status);
        }

        if(status != null && TF_GetCode(status) != TF_OK) {
            throw new IllegalStateException("ERROR: Unable to delete session " + TF_Message(status).getString());
        }



        if(status != null) {
            TF_DeleteStatus(status);
        }
    }
    public static org.tensorflow.framework.ConfigProto getAlignedWithNd4j() {
        org.tensorflow.framework.ConfigProto configProto = org.tensorflow.framework.ConfigProto.getDefaultInstance();
        ConfigProto.Builder builder1 = configProto.toBuilder().addDeviceFilters(TensorflowConversion.defaultDeviceForThread());
        try {
            //cuda
            if(Nd4j.getBackend().getClass().getName().toLowerCase().contains("jcu")) {
                builder1.setGpuOptions(org.tensorflow.framework.GPUOptions.newBuilder()
                        .setAllowGrowth(true)
                        .setPerProcessGpuMemoryFraction(0.5)
                        .build());
            }
            //cpu
            else {
            }

        } catch (Exception e) {
            log.error("",e);
        }

        return builder1.build();
    }



}
