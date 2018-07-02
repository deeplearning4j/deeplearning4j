package org.nd4j.tensorflow.conversion.graphrunner;

import com.github.os72.protobuf351.InvalidProtocolBufferException;
import com.github.os72.protobuf351.util.JsonFormat;
import com.google.protobuf.ByteString;
import lombok.Getter;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.tensorflow;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.TensorflowConversion;
import org.tensorflow.framework.NodeDef;

import java.io.Closeable;
import java.io.IOException;
import java.util.*;

import static org.bytedeco.javacpp.tensorflow.*;

/**
 * Runs a tensorflow session based on zero copy
 * {@link INDArray}
 *
 * @author Adam Gibson
 */
public class GraphRunner implements Closeable {
    //stored temporarily: this should  be null by end of initialization
    private byte[] graphToUse;
    //the in memory representation parsed from protobuf
    private tensorflow.TF_Graph graph;
    //the conversion between nd4j and tensorflow
    private TensorflowConversion conversion = new TensorflowConversion();
    //a persistent session to be used when running the graph
    private tensorflow.TF_Session session;
    //the options for the model
    private tensorflow.TF_SessionOptions options;
    //a status object used
    private tensorflow.TF_Status status;
    @Getter
    private Set<String> inputsForGraph,outputsForGraph;
    private List<String> inputOrder,outputOrder;
    @Getter
    private org.bytedeco.javacpp.tensorflow.ConfigProto sessionOptionsConfiguration;
    @Getter
    private org.tensorflow.framework.ConfigProto protoBufConfigProto;
    /**
     * Initialize with the graph content to use
     * @param graphToUse the raw byte content
     *                   of a protobuf file saved by tensorflow
     */
    public GraphRunner(byte[] graphToUse) {
        this.graphToUse = graphToUse;
        initSessionAndStatusIfNeeded();
    }

    /**
     * Initialize with the graph content to use
     * @param graphToUse the raw byte content
     *                   of a protobuf file saved by tensorflow
     * @param sessionOptionsConfiguration the session options to use
     *                                    for running sessions
     */
    public GraphRunner(byte[] graphToUse,org.bytedeco.javacpp.tensorflow.ConfigProto sessionOptionsConfiguration) {
        this.graphToUse = graphToUse;
        this.sessionOptionsConfiguration = sessionOptionsConfiguration;

        try {
            this.protoBufConfigProto = org.tensorflow.framework.ConfigProto.parseFrom(sessionOptionsConfiguration.SerializeAsString().getStringBytes());
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
            throw new IllegalArgumentException("Unable to parse protobuf",e);
        }

        initSessionAndStatusIfNeeded();
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



        if(inputOrder.size() != inputsForGraph.size()) {
            throw new IllegalArgumentException("Input order specified does not match inferred inputs from graph definition. Missing inputs?");
        }

        if(inputs.size() != inputOrder.size()) {
            throw new IllegalArgumentException("Number of inputs specified do not match number of arrays specified.");
        }


        Map<String,INDArray> outputArrays = new LinkedHashMap<>();

        Map<String,TF_Operation> opsByName = new HashMap<>();
        tensorflow.TF_Output inputOut = new tensorflow.TF_Output(inputOrder.size());

        TF_Tensor[] inputTensors = new TF_Tensor[inputOrder.size()];
        for(int i = 0; i < inputOrder.size(); i++) {
            tensorflow.TF_Operation inputOp = TF_GraphOperationByName(graph, inputOrder.get(i));
            opsByName.put(inputOrder.get(i),inputOp);
            inputOut.position(i).oper(inputOp).index(0);
            TF_Tensor tf_tensor = conversion.tensorFromNDArray(inputs.get(inputOrder.get(i)));
            inputTensors[i] = tf_tensor;
        }


        //reset the position of the pointer for execution
        inputOut.position(0);

        TF_Output outputOut = new tensorflow.TF_Output(outputOrder.size());
        //only setup the output ops
        for(int i = 0; i < outputOrder.size(); i++) {
            tensorflow.TF_Operation outputOp = TF_GraphOperationByName(graph, outputOrder.get(i));
            opsByName.put(outputOrder.get(i),outputOp);
            outputOut.position(i).oper(outputOp).position(i).index(0);
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
            throw new RuntimeException("ERROR: Unable to run session " + TF_Message(status).getString());
        } else {
            for(int i = 0; i < outputOrder.size(); i++) {
                INDArray to = conversion.ndArrayFromTensor(new TF_Tensor(outputTensorsPointer.get(i)));
                outputArrays.put(outputOrder.get(i),to);
            }

        }

        return outputArrays;
    }

    private void initSessionAndStatusIfNeeded() {
        try {
            //use the protobuf api to load the graph definition and load the node metadata
            org.tensorflow.framework.GraphDef graphDef1 = org.tensorflow.framework.GraphDef.parseFrom(graphToUse);
            inputsForGraph = new LinkedHashSet<>();
            outputsForGraph = new LinkedHashSet<>();
            //infer the inputs and outputs for the graph
            Set<String> seenAsInput = new LinkedHashSet<>();
            for(int i = 0; i < graphDef1.getNodeCount(); i++) {
                NodeDef node = graphDef1.getNode(i);
                if(node.getInputCount() < 1) {
                    inputsForGraph.add(node.getName());
                }

                for(int input = 0; input < node.getInputCount(); input++) {
                    seenAsInput.add(node.getInput(input));
                }
            }
            //find the nodes that were not inputs to any  nodes: these are the outputs
            for(int i = 0; i < graphDef1.getNodeCount(); i++) {
                if(!seenAsInput.contains(graphDef1.getNode(i).getName())) {
                    outputsForGraph.add(graphDef1.getNode(i).getName());
                }
            }

            //used for random access
            inputOrder = new ArrayList<>(inputsForGraph);
            outputOrder = new ArrayList<>(outputsForGraph);

        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }


        //setup the status object to be used for all tensorflow calls
        if(status == null) {
            status = TF_NewStatus();
        }


        //setup and configure the session, factoring
        //in the ConfigObject as needed
        if(session == null) {
            graph = conversion.getInitializedGraphForNd4jDevices(graphToUse);

            options = TF_NewSessionOptions();
            if(sessionOptionsConfiguration != null) {
                BytePointer bytePointer = sessionOptionsConfiguration.SerializeAsString();
                TF_SetConfig(options,bytePointer,bytePointer.getStringBytes().length,status);
                if (TF_GetCode(status) != TF_OK) {
                    throw new RuntimeException("ERROR: Unable to set value configuration:" + TF_Message(status).getString());
                }
            }

            session = tensorflow.TF_NewSession(graph, options, status);
            if (TF_GetCode(status) != TF_OK) {
                throw new RuntimeException("ERROR: Unable to open session " + TF_Message(status).getString());
            }

        }

        //get rid of the graph representation once used
        graphToUse = null;

    }


    /**
     * Convert a json string written out
     * by {@link com.github.os72.protobuf351.util.JsonFormat}
     * to a {@link org.bytedeco.javacpp.tensorflow.ConfigProto}
     * @param json the json to read
     * @return the config proto to use
     */
    public static org.tensorflow.framework.ConfigProto fromJson(String json) {
        org.tensorflow.framework.ConfigProto.Builder builder = org.tensorflow.framework.ConfigProto.newBuilder();
        try {
            com.google.protobuf.util.JsonFormat.parser().merge(json,builder);
            org.tensorflow.framework.ConfigProto build = builder.build();
            ByteString serialized = build.toByteString();
            byte[] binaryString = serialized.toByteArray();
            org.tensorflow.framework.ConfigProto configProto = org.tensorflow.framework.ConfigProto.parseFrom(binaryString);
            return configProto;
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
            e.printStackTrace();
        }

        return null;
    }


    /**
     * Write out the session options used
     * by this {@link GraphRunner}
     * a s a  json string using the
     * {@link com.google.protobuf.util.JsonFormat}
     * @return
     */
    public  String sessionOptionsToJson() {
        try {
            return JsonFormat.printer().print(sessionOptionsConfiguration);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
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
            throw new RuntimeException("ERROR: Unable to delete session " + TF_Message(status).getString());
        }



        if(status != null) {
            TF_DeleteStatus(status);
        }
    }
}
