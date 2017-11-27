package org.nd4j.imports;

import lombok.val;
import onnx.OnnxProto3;
import org.junit.Test;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

public class OnnxGraphMapperTests {

    @Test
    public void testMapper() throws Exception {
       try(val inputs = new ClassPathResource("onnx_graphs/embedding_only.onnx").getInputStream()) {
           OnnxProto3.GraphProto graphProto = OnnxProto3.ModelProto.parseFrom(inputs).getGraph();
           OnnxGraphMapper onnxGraphMapper = new OnnxGraphMapper();
           assertEquals(graphProto.getNodeList().size(),
                   onnxGraphMapper.getNodeList(graphProto).size());
           assertEquals(3,onnxGraphMapper.variablesForGraph(graphProto).size());
           val initializer = graphProto.getInput(0).getType().getTensorType();
           INDArray arr = onnxGraphMapper.getNDArrayFromTensor(graphProto.getInitializer(0).getName(), initializer, graphProto);
           assumeNotNull(arr);
           for(val node : graphProto.getNodeList()) {
               assertEquals(node.getAttributeList().size(),onnxGraphMapper.getAttrMap(node).size());
           }

           val sameDiff = onnxGraphMapper.importGraph(graphProto);
           assertEquals(3,sameDiff.graph().numVertices());
       }

    }


    @Test
    public void testLoadVgg16() throws Exception {
        val loadedFile = new ClassPathResource("onnx_graphs/vgg16/model.pb").getInputStream();

        val mapped = OnnxGraphMapper.getInstance().importGraph(loadedFile);
    }
}
