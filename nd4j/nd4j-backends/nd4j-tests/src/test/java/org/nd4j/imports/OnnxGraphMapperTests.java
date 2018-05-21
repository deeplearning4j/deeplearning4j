package org.nd4j.imports;

import lombok.val;
import onnx.OnnxProto3;
import org.junit.After;
import org.junit.Test;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

public class OnnxGraphMapperTests {

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Test
    public void testMapper() throws Exception {
        try(val inputs = new ClassPathResource("onnx_graphs/embedding_only.onnx").getInputStream()) {
            OnnxProto3.GraphProto graphProto = OnnxProto3.ModelProto.parseFrom(inputs).getGraph();
            OnnxGraphMapper onnxGraphMapper = new OnnxGraphMapper();
            assertEquals(graphProto.getNodeList().size(),
                    onnxGraphMapper.getNodeList(graphProto).size());
            assertEquals(4,onnxGraphMapper.variablesForGraph(graphProto).size());
            val initializer = graphProto.getInput(0).getType().getTensorType();
            INDArray arr = onnxGraphMapper.getNDArrayFromTensor(graphProto.getInitializer(0).getName(), initializer, graphProto);
            assumeNotNull(arr);
            for(val node : graphProto.getNodeList()) {
                assertEquals(node.getAttributeList().size(),onnxGraphMapper.getAttrMap(node).size());
            }

            val sameDiff = onnxGraphMapper.importGraph(graphProto);
            assertEquals(1,sameDiff.functions().length);
            System.out.println(sameDiff);
        }

    }

    @Test
    public void test1dCnn() throws Exception {
        val loadedFile = new ClassPathResource("onnx_graphs/sm_cnn.onnx").getInputStream();
        val mapped = OnnxGraphMapper.getInstance().importGraph(loadedFile);
        System.out.println(mapped.variables());
    }





    @Test
    public void testLoadResnet() throws Exception {
        val loadedFile = new ClassPathResource("onnx_graphs/resnet50/model.pb").getInputStream();

        val mapped = OnnxGraphMapper.getInstance().importGraph(loadedFile);
    }
}
