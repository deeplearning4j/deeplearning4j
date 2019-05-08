/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.tensorflow.conversion.graphrunner;

import com.github.os72.protobuf351.ByteString;
import com.github.os72.protobuf351.InvalidProtocolBufferException;
import com.github.os72.protobuf351.util.JsonFormat;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.tensorflow.conversion.TensorflowConversion;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;
import org.tensorflow.framework.NodeDef;

import java.io.Closeable;
import java.io.File;
import java.io.IOException;
import java.util.*;

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
public class GraphRunner implements Closeable {
    private SavedModelConfig savedModelConfig;
    //the in memory representation parsed from protobuf
    private TF_Graph graph;
    //the conversion between nd4j and tensorflow
    private TensorflowConversion conversion =  TensorflowConversion.getInstance();
    //a persistent session to be used when running the graph
    private TF_Session session;
    //the options for the model
    private TF_SessionOptions options;
    //a status object used
    private TF_Status status;
    @Getter
    @Setter
    private List<String> inputOrder,outputOrder;
    @Getter
    private org.tensorflow.framework.ConfigProto protoBufConfigProto;

    /**
     * Pass in a graph instance and
     * the length of the protobuf
     * that it was instantiated with.
     * For files this is typically
     * {@link File#length()},
     * for byte arrays, this is
     * byte array.length
     * and for {@link java.nio.ByteBuffer}
     * this would be something like the
     * {@link java.nio.ByteBuffer#capacity()}
     * @param inputNames  the input names for the graph
     * @param outputNames the output names in the graph
     * @param graph a pointer to the {@link TF_Graph} to use when executing
     * @param graphDef {@link org.tensorflow.framework.GraphDef} protobuf
     *                                                          definition containing
     *                                                          the graph configuration
     *                                                          for automatically inferring
     *                                                          things like
     *                                                          graph inputs and outputs
     *
     *
     */
    public GraphRunner(List<String> inputNames,List<String> outputNames,TF_Graph graph,org.tensorflow.framework.GraphDef graphDef) {
        this(inputNames,outputNames,graph,graphDef,null);

    }

    /**
     * Pass in a graph instance and
     * the length of the protobuf
     * that it was instantiated with.
     * For files this is typically
     * {@link File#length()},
     * for byte arrays, this is
     * byte array.length
     * and for {@link java.nio.ByteBuffer}
     * this would be something like the
     * {@link java.nio.ByteBuffer#capacity()}
     * @param graph a pointer to the {@link TF_Graph} to use when executing
     * @param graphDef {@link org.tensorflow.framework.GraphDef} protobuf
     *                                                          definition containing
     *                                                          the graph configuration
     *                                                          for automatically inferring
     *                                                          things like
     *                                                          graph inputs and outputs
     * @param configProto  the session configuration proto to use with this runner
     */
    public GraphRunner(List<String> inputNames,List<String> outputNames,TF_Graph graph,org.tensorflow.framework.GraphDef graphDef,ConfigProto configProto) {
        this.graph = graph;
        this.protoBufConfigProto = configProto;
        this.inputOrder = inputNames;
        this.outputOrder = outputNames;
        initSessionAndStatusIfNeeded(graphDef);

    }

    /**
     * Initialize with the graph content to use
     * @param inputNames the inputs to the graph
     * @param graphToUse the raw byte content
     *                   of a protobuf file saved by tensorflow
     */
    public GraphRunner(byte[] graphToUse,List<String> inputNames,List<String> outputNames) {
        this(graphToUse,inputNames,outputNames,getAlignedWithNd4j());
    }


    /**
     * Initialize with the graph content to use
     * @param filePath path of a protobuf file saved by tensorflow
     * @param inputNames the input namesfor the graph
     */
    public GraphRunner(String filePath,List<String> inputNames,List<String> outputNames) {
        this(filePath,inputNames,outputNames,getAlignedWithNd4j());
    }



    /**
     * Initialize with the graph content to use
     * @param filePath path of a protobuf file saved by tensorflow
     * @param inputNames the names of the inputs for the graph
     * @param sessionOptionsConfiguration the session options to use
     *                                    for running sessions
     */
    public GraphRunner(String filePath,List<String> inputNames,List<String> outputNames,org.tensorflow.framework.ConfigProto sessionOptionsConfiguration) {
        byte[] graphToUse = null;

        try {
            this.inputOrder = inputNames;
            this.outputOrder = outputNames;
            this.protoBufConfigProto = sessionOptionsConfiguration;
            initOptionsIfNeeded();
            graphToUse = IOUtils.toByteArray(new File(filePath).toURI());
            this.graph = conversion.loadGraph(graphToUse, status);
        } catch (Exception e) {
            throw new IllegalArgumentException("Unable to parse protobuf",e);
        }

        initSessionAndStatusIfNeeded(graphToUse);
    }

    /**
     * Initialize with the graph content to use
     * @param graphToUse the raw byte content
     *                   of a protobuf file saved by tensorflow
     * @param sessionOptionsConfiguration the session options to use
     *                                    for running sessions
     */
    public GraphRunner(byte[] graphToUse,List<String> inputNames,List<String> outputNames,org.tensorflow.framework.ConfigProto sessionOptionsConfiguration) {
        try {
            this.inputOrder = inputNames;
            this.outputOrder = outputNames;
            this.protoBufConfigProto = sessionOptionsConfiguration;
            initOptionsIfNeeded();
            this.graph = conversion.loadGraph(graphToUse, status);
        } catch (Exception e) {
            throw new IllegalArgumentException("Unable to parse protobuf",e);
        }

        initSessionAndStatusIfNeeded(graphToUse);
    }


    /**
     * Initialize with the SavedModel to use
     * @param inputNames (optional) the input names for the tensorflow graph
     * @param outputNames the output names for the tensorflow graph
     * @param savedModelConfig the configuration of the model to run
     */
    public GraphRunner(List<String> inputNames,List<String> outputNames,SavedModelConfig savedModelConfig) {
        this(inputNames,outputNames,savedModelConfig,getAlignedWithNd4j());
    }

    /**
     * Initialize with the SavedModel to use
     * @param inputNames (optional) the input names for the tensorflow graph
     * @param outputNames (optional) the output names for the tensorflow graph
     * @param savedModelConfig the configuration for the saved model
     * @param sessionOptionsConfiguration the session options to use
     *                                    for running sessions
     */
    public GraphRunner(List<String> inputNames,List<String> outputNames,SavedModelConfig savedModelConfig, ConfigProto sessionOptionsConfiguration) {
        try {
            this.savedModelConfig = savedModelConfig;
            this.protoBufConfigProto = sessionOptionsConfiguration;
            //note that the input and output order, maybe null here
            //if the names are specified, we should defer to those instead
            this.inputOrder = inputNames;
            this.outputOrder = outputNames;
            initOptionsIfNeeded();
            Map inputsMap = new LinkedHashMap<String, String>();
            Map outputsMap = new LinkedHashMap<String, String>();
            this.graph = TF_NewGraph();
            this.session = conversion.loadSavedModel(savedModelConfig, options, null, graph, inputsMap, outputsMap, status);
            inputOrder = new ArrayList<String>(inputsMap.keySet());
            outputOrder = new ArrayList<String>(outputsMap.keySet());
            savedModelConfig.setSavedModelInputOrder(new ArrayList<String>(inputsMap.values()));
            savedModelConfig.setSaveModelOutputOrder(new ArrayList<String>(outputsMap.values()));
        } catch (Exception e) {
            throw new IllegalArgumentException("Unable to parse protobuf",e);
        }
    }




    /**
     * Pass in a graph instance and
     * the length of the protobuf
     * that it was instantiated with.
     * For files this is typically
     * {@link File#length()},
     * for byte arrays, this is
     * byte array.length
     * and for {@link java.nio.ByteBuffer}
     * this would be something like the
     * {@link java.nio.ByteBuffer#capacity()}
     * @param graph a pointer to the {@link TF_Graph} to use when executing
     * @param graphDef {@link org.tensorflow.framework.GraphDef} protobuf
     *                                                          definition containing
     *                                                          the graph configuration
     *                                                          for automatically inferring
     *                                                          things like
     *                                                          graph inputs and outputs
     */
    public GraphRunner(List<String> inputNames,TF_Graph graph,org.tensorflow.framework.GraphDef graphDef) {
        this(inputNames,null,graph,graphDef,null);

    }

    /**
     * Pass in a graph instance and
     * the length of the protobuf
     * that it was instantiated with.
     * For files this is typically
     * {@link File#length()},
     * for byte arrays, this is
     * byte array.length
     * and for {@link java.nio.ByteBuffer}
     * this would be something like the
     * {@link java.nio.ByteBuffer#capacity()}
     * @param graph a pointer to the {@link TF_Graph} to use when executing
     * @param graphDef {@link org.tensorflow.framework.GraphDef} protobuf
     *                                                          definition containing
     *                                                          the graph configuration
     *                                                          for automatically inferring
     *                                                          things like
     *                                                          graph inputs and outputs
     * @param configProto  the session configuration proto to use with this runner
     */
    public GraphRunner(List<String> inputNames,TF_Graph graph,org.tensorflow.framework.GraphDef graphDef,ConfigProto configProto) {
        this(inputNames,null,graph,graphDef,configProto);

    }

    /**
     * Initialize with the graph content to use
     * @param inputNames the inputs to the graph
     * @param graphToUse the raw byte content
     *                   of a protobuf file saved by tensorflow
     */
    public GraphRunner(byte[] graphToUse,List<String> inputNames) {
        this(graphToUse,inputNames,getAlignedWithNd4j());
    }


    /**
     * Initialize with the graph content to use
     * @param filePath path of a protobuf file saved by tensorflow
     * @param inputNames the input namesfor the graph
     */
    public GraphRunner(String filePath,List<String> inputNames) {
        this(filePath,inputNames,getAlignedWithNd4j());
    }



    /**
     * Initialize with the graph content to use
     * @param filePath path of a protobuf file saved by tensorflow
     * @param inputNames the names of the inputs for the graph
     * @param sessionOptionsConfiguration the session options to use
     *                                    for running sessions
     */
    public GraphRunner(String filePath,List<String> inputNames,org.tensorflow.framework.ConfigProto sessionOptionsConfiguration) {
        this(filePath,inputNames,null,sessionOptionsConfiguration);
    }

    /**
     * Initialize with the graph content to use
     * @param graphToUse the raw byte content
     *                   of a protobuf file saved by tensorflow
     * @param sessionOptionsConfiguration the session options to use
     *                                    for running sessions
     */
    public GraphRunner(byte[] graphToUse,List<String> inputNames,org.tensorflow.framework.ConfigProto sessionOptionsConfiguration) {
        this(graphToUse,inputNames,null,sessionOptionsConfiguration);
    }


    /**
     * Initialize with the SavedModel to use
     * @param savedModelConfig the configuration for loading the saved model
     */
    public GraphRunner(SavedModelConfig savedModelConfig) {
        this(savedModelConfig,getAlignedWithNd4j());
    }

    /**
     * Initialize with the SavedModel to use
     * @param savedModelConfig the configuration for loading the saved model
     * @param sessionOptionsConfiguration the session options to use
     *                                    for running sessions
     */
    public GraphRunner(SavedModelConfig savedModelConfig, ConfigProto sessionOptionsConfiguration) {
        try {
            this.savedModelConfig = savedModelConfig;
            this.protoBufConfigProto = sessionOptionsConfiguration;
            initOptionsIfNeeded();
            Map<String,String> inputsMap = new LinkedHashMap<>();
            Map<String,String> outputsMap = new LinkedHashMap<>();
            this.graph = TF_NewGraph();
            this.session = conversion.loadSavedModel(savedModelConfig, options, null, graph, inputsMap, outputsMap, status);
            inputOrder = new ArrayList<>(inputsMap.keySet());
            outputOrder = new ArrayList<>(outputsMap.keySet());
            savedModelConfig.setSavedModelInputOrder(new ArrayList<>(inputsMap.values()));
            savedModelConfig.setSaveModelOutputOrder(new ArrayList<>(outputsMap.values()));
        } catch (Exception e) {
            throw new IllegalArgumentException("Unable to parse protobuf",e);
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
     * @throws IOException
     */
    public Map<String,INDArray> run(Map<String,INDArray> inputs) {
        if(graph == null) {
            throw new IllegalStateException("Graph not initialized.");
        }

        if(inputs.size() != inputOrder.size()) {
            throw new IllegalArgumentException("Number of inputs specified do not match number of arrays specified.");
        }


        if(savedModelConfig != null) {
            Map<String,INDArray> outputArrays = new LinkedHashMap<>();

            Map<String,TF_Operation> opsByName = new HashMap<>();
            TF_Output inputOut = new TF_Output(savedModelConfig.getSavedModelInputOrder().size());

            TF_Tensor[] inputTensors = new TF_Tensor[savedModelConfig.getSavedModelInputOrder().size()];
            for(int i = 0; i < savedModelConfig.getSavedModelInputOrder().size(); i++) {
                String[] name = savedModelConfig.getSavedModelInputOrder().get(i).split(":");
                TF_Operation inputOp = TF_GraphOperationByName(graph, name[0]);
                opsByName.put(savedModelConfig.getSavedModelInputOrder().get(i),inputOp);
                inputOut.position(i).oper(inputOp).index(name.length > 1 ? Integer.parseInt(name[1]) : 0);
                TF_Tensor tf_tensor = conversion.tensorFromNDArray(inputs.get(inputOrder != null && !inputOrder.isEmpty()
                        ? inputOrder.get(i) : savedModelConfig.getSavedModelInputOrder().get(i)));
                inputTensors[i] = tf_tensor;
            }


            //reset the position of the pointer for execution
            inputOut.position(0);

            TF_Output outputOut = new TF_Output(savedModelConfig.getSaveModelOutputOrder().size());
            //only setup the output ops
            for(int i = 0; i < savedModelConfig.getSaveModelOutputOrder().size(); i++) {
                String[] name =savedModelConfig.getSaveModelOutputOrder().get(i).split(":");
                TF_Operation outputOp = TF_GraphOperationByName(graph, name[0]);
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


            TF_SessionRun(
                    session,
                    null,
                    //inputs
                    inputOut, inputTensorsPointer, inputTensors.length,
                    //outputs
                    outputOut, outputTensorsPointer, savedModelConfig.getSaveModelOutputOrder().size(),
                    //targets
                    null, 0,
                    null,
                    status);


            if (TF_GetCode(status) != TF_OK) {
                throw new IllegalStateException("ERROR: Unable to run session " + TF_Message(status).getString());
            } else {
                for(int i = 0; i < outputOrder.size(); i++) {
                    INDArray to = conversion.ndArrayFromTensor(new TF_Tensor(outputTensorsPointer.get(i)));
                    outputArrays.put(outputOrder != null && !outputOrder.isEmpty() ? outputOrder.get(i) :
                            savedModelConfig.getSaveModelOutputOrder().get(i),to);
                }

            }

            return outputArrays;

        }
        else {
            Map<String,INDArray> outputArrays = new LinkedHashMap<>();

            Map<String,TF_Operation> opsByName = new HashMap<>();
            TF_Output inputOut = new TF_Output(inputOrder.size());

            TF_Tensor[] inputTensors = new TF_Tensor[inputOrder.size()];
            for(int i = 0; i < inputOrder.size(); i++) {
                String[] name = inputOrder.get(i).split(":");
                TF_Operation inputOp = TF_GraphOperationByName(graph, name[0]);
                opsByName.put(inputOrder.get(i),inputOp);
                inputOut.position(i).oper(inputOp).index(name.length > 1 ? Integer.parseInt(name[1]) : 0);
                TF_Tensor tf_tensor = conversion.tensorFromNDArray(inputs.get(inputOrder.get(i)));
                inputTensors[i] = tf_tensor;
            }


            //reset the position of the pointer for execution
            inputOut.position(0);

            TF_Output outputOut = new TF_Output(outputOrder.size());
            //only setup the output ops
            for(int i = 0; i < outputOrder.size(); i++) {
                String[] name = outputOrder.get(i).split(":");
                TF_Operation outputOp = TF_GraphOperationByName(graph, name[0]);
                if(outputOp == null) {
                    throw new IllegalArgumentException("Illegal input found " + inputOrder.get(i) + " - no op found! Mis specified name perhaps?");
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


            TF_SessionRun(
                    session,
                    null,
                    //inputs
                    inputOut, inputTensorsPointer, inputTensors.length,
                    //outputs
                    outputOut, outputTensorsPointer, outputOrder.size(),
                    //targets
                    null, 0,
                    null,
                    status);


            if (TF_GetCode(status) != TF_OK) {
                throw new IllegalStateException("ERROR: Unable to run session " + TF_Message(status).getString());
            } else {
                for(int i = 0; i < outputOrder.size(); i++) {
                    INDArray to = conversion.ndArrayFromTensor(new TF_Tensor(outputTensorsPointer.get(i)));
                    outputArrays.put(outputOrder.get(i),to);
                }

            }

            return outputArrays;

        }

    }


    private void initOptionsIfNeeded() {
        //setup the status object to be used for all tensorflow calls
        if(status == null) {
            status = TF_NewStatus();
        }

        if (options == null) {
            options = TF_NewSessionOptions();
            if(protoBufConfigProto != null) {
                BytePointer bytePointer = new BytePointer(protoBufConfigProto.toByteArray());
                TF_SetConfig(options,bytePointer,bytePointer.getStringBytes().length,status);
                if (TF_GetCode(status) != TF_OK) {
                    throw new IllegalStateException("ERROR: Unable to set value configuration:" + TF_Message(status).getString());
                }
            }
        }
    }

    private void initSessionAndStatusIfNeeded(org.tensorflow.framework.GraphDef graphDef1) {
        //infer the inputs and outputs for the graph
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
            //find the nodes that were not inputs to any  nodes: these are the outputs
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
        try {
            //use the protobuf api to load the graph definition and load the node metadata
            org.tensorflow.framework.GraphDef graphDef1 = org.tensorflow.framework.GraphDef.parseFrom(graphToUse);
            initSessionAndStatusIfNeeded(graphDef1);
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
    }

    public static org.tensorflow.framework.ConfigProto getAlignedWithNd4j() {
        org.tensorflow.framework.ConfigProto configProto = org.tensorflow.framework.ConfigProto.getDefaultInstance();
        ConfigProto.Builder builder1 = configProto.toBuilder().addDeviceFilters(TensorflowConversion.defaultDeviceForThread());
        try {
            //cuda
            if(Nd4j.getBackend().getClass().getName().toLowerCase().contains("jcu")) {
                builder1.setGpuOptions(GPUOptions.newBuilder()
                        .setAllowGrowth(true)
                        .setPerProcessGpuMemoryFraction(0.5)
                        .build());
            }
            //cpu
            else {
            }

        } catch (Exception e) {
            e.printStackTrace();
        }

        return builder1.build();
    }


    /**
     * Convert a json string written out
     * by {@link com.github.os72.protobuf351.util.JsonFormat}
     * to a {@link org.bytedeco.tensorflow.ConfigProto}
     * @param json the json to read
     * @return the config proto to use
     */
    public static org.tensorflow.framework.ConfigProto fromJson(String json) {
        org.tensorflow.framework.ConfigProto.Builder builder = org.tensorflow.framework.ConfigProto.newBuilder();
        try {
            JsonFormat.parser().merge(json,builder);
            org.tensorflow.framework.ConfigProto build = builder.build();
            ByteString serialized = build.toByteString();
            byte[] binaryString = serialized.toByteArray();
            org.tensorflow.framework.ConfigProto configProto = org.tensorflow.framework.ConfigProto.parseFrom(binaryString);
            return configProto;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }


    /**
     * Write out the session options used
     * by this {@link GraphRunner}
     * a s a  json string using the
     * {@link JsonFormat}
     * @return
     */
    public  String sessionOptionsToJson() {
        try {
            return JsonFormat.printer().print(protoBufConfigProto);
        } catch (Exception e) {
            e.printStackTrace();
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
}
