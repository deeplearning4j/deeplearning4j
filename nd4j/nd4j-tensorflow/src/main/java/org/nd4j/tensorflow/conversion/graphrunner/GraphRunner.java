package org.nd4j.tensorflow.conversion.graphrunner;

import com.github.os72.protobuf351.InvalidProtocolBufferException;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.tensorflow;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.TensorflowConversion;
import org.tensorflow.framework.GraphDef;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.*;

public class GraphRunner implements Closeable {

    private byte[] graphToUse;
    private tensorflow.TF_Graph graph;
    private TensorflowConversion conversion = new TensorflowConversion();
    private tensorflow.TF_Session session;
    private tensorflow.TF_Status status;
    private List<String> opNames;

    public GraphRunner(byte[] graphToUse) {
        this.graphToUse = graphToUse;
    }

    public List<String> loadOpNames() throws InvalidProtocolBufferException {
        if(this.opNames != null)
            return this.opNames;

        GraphDef graphDef1 = GraphDef.parseFrom(graphToUse);
        List<String>  opNames = new ArrayList<>();
        for(int i = 0; i < graphDef1.getNodeCount(); i++) {
            opNames.add(graphDef1.getNode(i).getName());
        }

        this.opNames = opNames;
        return opNames;
    }

    public void run(Map<String,INDArray> inputs,
                    List<String> inputOrder,
                    Map<String,INDArray> outputArrays,
                    List<String> outputOrder) throws IOException {
        if(graph == null) {
            graph = conversion.getInitializedGraphForNd4jDevices(graphToUse);
        }

        if(status == null) {
            status = TF_NewStatus();
        }


        if(session == null) {
            tensorflow.TF_SessionOptions options = TF_NewSessionOptions();
            session = tensorflow.TF_NewSession(graph, options, status);
            if (TF_GetCode(status) != TF_OK) {
                throw new RuntimeException("ERROR: Unable to open session " + TF_Message(status).getString());
            }

            TF_DeleteSessionOptions(options);
        }



        tensorflow.TF_Output inputOut = new tensorflow.TF_Output(inputOrder.size());

        tensorflow.TF_Output outputOut = new tensorflow.TF_Output(outputOrder.size());

        List<TF_Tensor> inputTensors = new ArrayList<>();

        for(int i = 0; i < inputOrder.size(); i++) {
            tensorflow.TF_Operation inputOp = TF_GraphOperationByName(graph, inputOrder.get(i));
            inputOut.oper(inputOp).index(i);
            inputTensors.add(conversion.tensorFromNDArray(inputs.get(inputOrder.get(i))));
        }


        List<TF_Tensor> outputTensors = new ArrayList<>();
        for(int i = 0; i < outputOrder.size(); i++) {
            tensorflow.TF_Operation outputOp = TF_GraphOperationByName(graph, outputOrder.get(i));
            outputOut.oper(outputOp).index(i);
            outputTensors.add(conversion.tensorFromNDArray(outputArrays.get(outputOrder.get(i))));
        }

        PointerPointer<TF_Tensor> inputTensorsPointer = new PointerPointer<>(inputTensors.toArray(new TF_Tensor[inputTensors.size()]));
        PointerPointer<TF_Tensor> outputTensorsPointer = new PointerPointer<>(outputTensors.toArray(new TF_Tensor[inputTensors.size()]));
        List<String> opNames = loadOpNames();
        List<TF_Operation> ops = new ArrayList<>();
        for(int i = 0; i < opNames.size(); i++) {
            ops.add(TF_GraphOperationByName(graph,opNames.get(i)));
        }

        PointerPointer<TF_Operation> operationPointerPointer = new PointerPointer<>(ops.toArray(new TF_Operation[ops.size()]));

        TF_SessionRun(session,
                null,
                inputOut,
                inputTensorsPointer,
                inputTensors.size(),
                outputOut,
                outputTensorsPointer
                ,outputTensors.size(),
                operationPointerPointer
                , ops.size(),
                null,
                status);


    }

    @Override
    public void close() throws IOException {
        TF_DeleteSession(session,status);
        if (TF_GetCode(status) != TF_OK) {
            throw new RuntimeException("ERROR: Unable to delete session " + TF_Message(status).getString());
        }

        if(status != null) {
            TF_DeleteStatus(status);
        }
    }
}
