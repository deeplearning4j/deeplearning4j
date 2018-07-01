package org.nd4j.tensorflow.conversion;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.tensorflow;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.tensorflow.framework.GraphDef;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class TensorflowConversionTest {


    @Test
    public void testConversionFromNdArray() throws Exception {
        INDArray arr = Nd4j.linspace(1,4,4);
        TensorflowConversion tensorflowConversion = new TensorflowConversion();
        tensorflow.TF_Tensor tf_tensor = tensorflowConversion.tensorFromNDArray(arr);
        INDArray fromTensor = tensorflowConversion.ndArrayFromTensor(tf_tensor);
        assertEquals(arr,fromTensor);
        arr.addi(1.0);
        tf_tensor = tensorflowConversion.tensorFromNDArray(arr);
        fromTensor = tensorflowConversion.ndArrayFromTensor(tf_tensor);
        assertEquals(arr,fromTensor);


    }

    @Test
    public void testCudaIfAvailable() throws Exception {
        TensorflowConversion tensorflowConversion = new TensorflowConversion();
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());
        //byte[] content = Files.readAllBytes(Paths.get(new File("/home/agibsonccc/code/dl4j-test-resources/src/main/resources/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").toURI()));
        tensorflow.TF_Graph initializedGraphForNd4jDevices = tensorflowConversion.getInitializedGraphForNd4jDevices(content);
        assertNotNull(initializedGraphForNd4jDevices);

        String deviceName = tensorflowConversion.defaultDeviceForThread();
        if(Nd4j.getBackend().getClass().getName().toLowerCase().contains("jcu")) {
            assertEquals("/gpu:0",deviceName);
        }
        else {
            assertEquals("/cpu:0",deviceName);
        }

        byte[] content2 = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());
        byte[] content3 = tensorflowConversion.setDeviceForGraphDef(content2);
        GraphDef graphDef1 = GraphDef.parseFrom(content3);
        for(int i = 0; i < graphDef1.getNodeCount(); i++)
            assertEquals(deviceName,graphDef1.getNode(i).getDevice());
        System.out.println(graphDef1);
    }




}
