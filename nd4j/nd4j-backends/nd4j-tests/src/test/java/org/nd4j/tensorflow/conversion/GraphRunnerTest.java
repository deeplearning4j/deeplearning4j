package org.nd4j.tensorflow.conversion;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.*;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.tensorflow.framework.GraphDef;

import java.util.*;

import static org.bytedeco.javacpp.tensorflow.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class GraphRunnerTest {

    tensorflow.TF_Operation PlaceHolder(tensorflow.TF_Graph  graph, tensorflow.TF_Status  status,int dtype, String name) {
        TF_OperationDescription desc = TF_NewOperation(graph, "Placeholder", name);
        TF_SetAttrType(desc, "dtype", TF_FLOAT);
        TF_Operation ret = TF_FinishOperation(desc, status);
        assertEquals(TF_OK,TF_GetCode(status));
        return ret;
    }

    tensorflow.TF_Operation Const(tensorflow.TF_Graph graph, tensorflow.TF_Status status, tensorflow.TF_Tensor  tensor, String name) {
        tensorflow.TF_OperationDescription desc = TF_NewOperation(graph, "Const", name);
        TF_SetAttrTensor(desc, "value", tensor, status);
        assertEquals(TF_OK,TF_GetCode(status));
        TF_SetAttrType(desc, "dtype", TF_TensorType(tensor));
        TF_Operation ret =  TF_FinishOperation(desc, status);
        assertEquals(TF_OK,TF_GetCode(status));
        return ret;
    }

    TF_Operation  Add(TF_Graph  graph, TF_Status  status, TF_Operation  one, TF_Operation  two,String name) {
        TF_OperationDescription desc = TF_NewOperation(graph, "AddN", name);
        TF_Output add_inputs = new TF_Output(2);
        add_inputs.position(0).oper(one).index(0);
        add_inputs.position(1).oper(two).index(0);
        add_inputs.position(0);
        TF_AddInputList(desc, add_inputs, 2);
        TF_Operation ret = TF_FinishOperation(desc, status);
        assertEquals(TF_OK,TF_GetCode(status));
        return ret;
    }

    @Test
    public void testStandalone() {
        TF_Graph  graph = TF_NewGraph();
        TF_SessionOptions  options = TF_NewSessionOptions();
        TF_Status  status = TF_NewStatus();
        TF_Session session = TF_NewSession(graph, options, status);

        float in_val_one = 4.0f;
        DataBuffer floatBufferOne = Nd4j.createBuffer(new float[]{in_val_one});
        float const_two = 2.0f;
        DataBuffer floatBufferTwo = Nd4j.createBuffer(new float[]{const_two});
        DataBuffer outBuffer = Nd4j.createBuffer(new float[]{-1});

        TF_Tensor  tensor_in =  TF_NewTensor(TF_FLOAT,new long[]{1},0,floatBufferOne.pointer(),4L,DummyDeAllocator.getInstance(),null);
        TF_Tensor  tensor_const_two = TF_NewTensor(TF_FLOAT,new long[]{1},0,floatBufferTwo.pointer(),4L,DummyDeAllocator.getInstance(),null);
        TF_Tensor  tensor_out =  TF_NewTensor(TF_FLOAT,new long[]{1},0,outBuffer.pointer(),4L,DummyDeAllocator.getInstance(),null);
        System.out.printf("Output Tensor Type: %d\n", TF_TensorType(tensor_out));

        // Operations
        TF_Operation  feed = PlaceHolder(graph, status, TF_FLOAT, "feed");
        TF_Operation  two = Const(graph, status, tensor_const_two, "const");
        TF_Operation  add = Add(graph, status, feed, two, "add");

        // Session Inputs
        TF_Output input_operations = new TF_Output(1 );
        input_operations.position(0).oper(feed).index(0);

        input_operations.position(0);

        PointerPointer<TF_Tensor> input_tensors = new PointerPointer<>(tensor_in,tensor_const_two);
        input_tensors.position(0);
        // Session Outputs
        TF_Output output_operations  = new TF_Output().oper(add).index(0);
        output_operations.position(0);
        PointerPointer<TF_Tensor> output_tensors = new PointerPointer<>(1);
        output_tensors.position(0);

        PointerPointer<TF_Operation> ops = new PointerPointer<>(add);
        ops.position(0);
        TF_SessionRun(session, null,
                // Inputs
                input_operations, input_tensors, 1,
                // Outputs
                output_operations, output_tensors, 1,
                // Target operations
                null, 0, null,
                status);

        assertEquals(TF_Message(status).getString(),TF_OK,TF_GetCode(status));
        assertEquals(TF_FLOAT,TF_TensorType(tensor_out));
        System.out.printf("Session Run Status: %d - %s\n",TF_GetCode(status), TF_Message(status));
        TF_Tensor result = new TF_Tensor(output_tensors.position(0).get());
        TensorflowConversion tensorflowConversion = new TensorflowConversion();
        INDArray arr = tensorflowConversion.ndArrayFromTensor(result);
        System.out.printf("Output Tensor Type: %d\n", TF_TensorType(new TF_Tensor(output_tensors.position(0).get())));
        Pointer outval = TF_TensorData(tensor_out).capacity(4);
        System.out.printf("Output Tensor Value: %.2f\n", outval.asByteBuffer().asFloatBuffer().get());

        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);

        TF_DeleteSessionOptions(options);

        TF_DeleteGraph(graph);

        TF_DeleteTensor(tensor_in);
        TF_DeleteTensor(tensor_out);
        TF_DeleteTensor(tensor_const_two);

        TF_DeleteStatus(status);
    }

    @Test
    public void testGraphRunner() throws Exception {
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());
        try(GraphRunner graphRunner = new GraphRunner(content)) {
            INDArray input1 = Nd4j.linspace(1,4,4).reshape(4);
            INDArray input2 = Nd4j.linspace(1,4,4).reshape(4);
            INDArray result = Nd4j.create(input1.shape());
            Map<String,INDArray> inputs = new LinkedHashMap<>();
            inputs.put("input_0",input1);
            inputs.put("input_1",input2);

            Map<String,INDArray> outputs = new HashMap<>();
            outputs.put("output",result);

            graphRunner.run(inputs, Arrays.asList("input_0","input_1"),outputs,Arrays.asList("output"));

            List<String> tensorflowDeviceList = graphRunner.getTensorflowDeviceList();
            assertTrue(!tensorflowDeviceList.isEmpty());


            INDArray assertion = input1.add(input2);
            assertEquals(assertion,result);
        }
    }

}
