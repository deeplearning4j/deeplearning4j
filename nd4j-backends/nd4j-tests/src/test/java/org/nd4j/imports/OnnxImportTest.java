package org.nd4j.imports;

import org.junit.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.linalg.io.ClassPathResource;

public class OnnxImportTest {

    @Test
    public void testOnnxImportEmbedding() throws Exception {
        OnnxGraphMapper onnxGraphMapper = new OnnxGraphMapper();
        SameDiff importGraph = onnxGraphMapper.importGraph(new ClassPathResource("onnx_graphs/embedding_only.onnx").getFile());
    }

    @Test
    public void testOnnxImportCnn() throws Exception {
        OnnxGraphMapper onnxGraphMapper = new OnnxGraphMapper();
        SameDiff importGraph = onnxGraphMapper.importGraph(new ClassPathResource("onnx_graphs/sm_cnn.onnx").getFile());
    }

}
