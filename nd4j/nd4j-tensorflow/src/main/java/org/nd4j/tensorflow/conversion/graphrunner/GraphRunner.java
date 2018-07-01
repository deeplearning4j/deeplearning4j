package org.nd4j.tensorflow.conversion.graphrunner;

import com.github.os72.protobuf351.InvalidProtocolBufferException;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.tensorflow.conversion.TensorflowConversion;
import org.tensorflow.framework.GraphDef;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.*;

public class GraphRunner implements Closeable {

    private byte[] graphToUse;
    private tensorflow.TF_Graph graph;
    private TensorflowConversion conversion = new TensorflowConversion();
    private tensorflow.TF_Session session;
    private TF_DeviceList tf_deviceList;
    private tensorflow.TF_SessionOptions options;
    private tensorflow.TF_Status status;
    private Pair<List<String>,List<String>> opNames;

    public GraphRunner(byte[] graphToUse) {
        this.graphToUse = graphToUse;
    }

    public Pair<List<String>,List<String>> loadOpNames() throws InvalidProtocolBufferException {
        if(this.opNames != null)
            return this.opNames;

        GraphDef graphDef1 = GraphDef.parseFrom(graphToUse);
        List<String>  opNames = new ArrayList<>();
        List<String> ops = new ArrayList<>();
        for(int i = 0; i < graphDef1.getNodeCount(); i++) {
            opNames.add(graphDef1.getNode(i).getName());
            ops.add(graphDef1.getNode(i).getOp());
        }

        this.opNames = Pair.of(opNames,ops);
        return this.opNames;
    }

    public void run(Map<String,INDArray> inputs,
                    List<String> inputOrder,
                    Map<String,INDArray> outputArrays,
                    List<String> outputOrder) throws IOException {
        if(graph == null) {
            graph = conversion.getInitializedGraphForNd4jDevices(graphToUse);
        }


        initSessionAndStatusIfNeeded();




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
        TF_Tensor[] outputTensors = new TF_Tensor[outputOrder.size()];
        for(int i = 0; i < outputOrder.size(); i++) {
            tensorflow.TF_Operation outputOp = TF_GraphOperationByName(graph, outputOrder.get(i));
            opsByName.put(outputOrder.get(i),outputOp);
            outputOut.position(i).oper(outputOp).position(i).index(0);
            TF_Tensor to = conversion.tensorFromNDArray(outputArrays.get(outputOrder.get(i)));
            outputTensors[i] = to;
        }

        //reset the position of the pointer for execution
        outputOut.position(0);



        PointerPointer<TF_Tensor> inputTensorsPointer = new PointerPointer<>(inputTensors).position(0);
        PointerPointer<TF_Tensor> outputTensorsPointer = new PointerPointer<>(outputTensors).position(0);


        TF_SessionRun(
                session,
                null,
                inputOut, inputTensorsPointer, inputTensors.length,
                outputOut, outputTensorsPointer, outputTensors.length,
                null
                , 0,
                null,
                status);


        if (TF_GetCode(status) != TF_OK) {
            throw new RuntimeException("ERROR: Unable to run session " + TF_Message(status).getString());
        } else {
            Pointer pointer = TF_TensorData(outputTensors[0]).capacity(4);
            FloatPointer floatPointer = new FloatPointer(pointer);
            FloatIndexer floatIndexer = FloatIndexer.create(floatPointer);

            Pointer input1Pointer = TF_TensorData(inputTensors[0]).capacity(4);
            FloatPointer floatPointer2 = new FloatPointer(input1Pointer);
            FloatIndexer floatIndexer2 = FloatIndexer.create(floatPointer2);


            Pointer input1Pointer3 = TF_TensorData(inputTensors[1]).capacity(4);
            FloatPointer floatPointer3 = new FloatPointer(input1Pointer3);
            FloatIndexer floatIndexer3 = FloatIndexer.create(floatPointer3);

            System.out.println(TF_Message(status).getString());
        }

    }

    private void initSessionAndStatusIfNeeded() {
        if(status == null) {
            status = TF_NewStatus();
        }


        if(session == null) {
            options = TF_NewSessionOptions();
            session = tensorflow.TF_NewSession(graph, options, status);
            if (TF_GetCode(status) != TF_OK) {
                throw new RuntimeException("ERROR: Unable to open session " + TF_Message(status).getString());
            }

        }

        if(tf_deviceList == null) {
            tf_deviceList = TF_SessionListDevices(session, status);
        }
    }

    private String getOpName(TF_Operation operation) {
        ByteBuffer byteBuffer = operation.asByteBuffer();
        byte[] ret = new byte[byteBuffer.capacity()];
        byteBuffer.get(ret);
        return new String(ret);
    }


    /**
     * Mainly used for debugging:
     * Checks the number of devices for tensorflow
     * @return
     */
    public int getNumDevicesForTensorflow() {
        initSessionAndStatusIfNeeded();
        int count = TF_DeviceListCount(tf_deviceList);
        return count;
    }


    public List<String> getTensorflowDeviceList() {
        initSessionAndStatusIfNeeded();
        List<String> ret = new ArrayList<>();
        int devices = getNumDevicesForTensorflow();
        for(int i = 0; i < devices; i++) {
            BytePointer bytePointer = TF_DeviceListName(tf_deviceList, i, status);
            if(status != null && TF_GetCode(status) != TF_OK) {
                throw new RuntimeException("ERROR: Unable to obtain name for device " + TF_Message(status).getString());
            }

            ret.add(new String(bytePointer.getStringBytes()));
        }

        return ret;
    }





    @Override
    public void close() throws IOException {
        if(status != null && tf_deviceList != null) {
            TF_DeleteDeviceList(tf_deviceList);
        }

        if(status != null && TF_GetCode(status) != TF_OK) {
            throw new RuntimeException("ERROR: Unable to delete device list " + TF_Message(status).getString());
        }


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
