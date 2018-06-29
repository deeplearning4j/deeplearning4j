package org.nd4j.tensorflow.conversion.org.nd4j.tensorflow.graphrunner;

import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.tensorflow;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.tensorflow.conversion.TensorflowConversion;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.*;

public class GraphRunner {

    private Map<String,INDArray> inputs;
    private byte[] graphToUse;
    private tensorflow.TF_Graph graph;
    private TensorflowConversion conversion = new TensorflowConversion();
    private tensorflow.TF_Session session;
    private tensorflow.TF_Status status;

    public void run(Map<String,INDArray> inputs,
                    List<String> inputOrder,
                    Map<String,INDArray> outputArrays,
                    List<String> outputOrder) throws IOException {
        tensorflow.TF_Graph graph = conversion.getInitializedGraphForNd4jDevices(graphToUse);
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

        tensorflow.TF_Operation inputOp = TF_GraphOperationByName(graph, inputOrder.get(0));
        tensorflow.TF_Output inputOut = new tensorflow.TF_Output().oper(inputOp).index(0);

        tensorflow.TF_Operation outputOp = TF_GraphOperationByName(graph, outputOrder.get(0));
        tensorflow.TF_Output outputOut = new tensorflow.TF_Output().oper(outputOp).index(0);
        List<TF_Operation> inputsOps = new ArrayList<>();
        List<TF_Tensor> inputTensors = new ArrayList<>();
        for(int i = 0; i < inputOrder.size(); i++) {
            inputTensors.add(conversion.tensorFromNDArray(inputs.get(inputOrder.get(i))));
            inputsOps.add(TF_GraphOperationByName(graph, inputOrder.get(i)));
        }


        List<TF_Tensor> outputTensors = new ArrayList<>();
        List<TF_Operation> outputOps = new ArrayList<>();
        for(int i = 0; i < outputOps.size(); i++) {
            outputTensors.add(conversion.tensorFromNDArray(outputArrays.get(outputOrder.get(i))));
            outputOps.add(TF_GraphOperationByName(graph, outputOrder.get(i)));
        }

        PointerPointer<TF_Tensor> inputTensorsPointer = new PointerPointer<>(inputTensors.toArray(new TF_Tensor[inputTensors.size()]));
        PointerPointer<TF_Tensor> outputTensorsPointer = new PointerPointer<>(outputTensors.toArray(new TF_Tensor[inputTensors.size()]));
        PointerPointer<TF_Operation> opsToRun = new PointerPointer<>(outputOps.toArray(new TF_Operation[outputOps.size()]));
        PointerPointer<TF_Operation> inputsToRun = new PointerPointer<>(inputsOps.toArray(new TF_Operation[outputOps.size()]));




        TF_SessionRun(session,
                null,
                inputOut,
                inputTensorsPointer,
                inputTensors.size(),
                outputOut,
                outputTensorsPointer
                ,outputTensors.size(),
                opsToRun
                , 0,
                null,
                status);


    }

}
