package org.nd4j.imports;

import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.assertArrayEquals;

public class OnnxImportTest {

    @After
    public void tearDown() throws Exception {
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Test
    public void testOnnxImportEmbedding() throws Exception {
        /**
         *
         */
        val importGraph = OnnxGraphMapper.getInstance().importGraph(new ClassPathResource("onnx_graphs/embedding_only.onnx").getInputStream());
        val embeddingMatrix = importGraph.getVariable("2");
        assertArrayEquals(new long[] {100,300},embeddingMatrix.getShape());
       /* val onlyOp = importGraph.getFunctionForVertexId(importGraph.getVariable("3").getVertexId());
        assertNotNull(onlyOp);
        assertTrue(onlyOp instanceof Gather);
*/
    }

    @Test
    public void testOnnxImportCnn() throws Exception {
   /*     val importGraph = OnnxGraphMapper.getInstance().importGraph(new ClassPathResource("onnx_graphs/sm_cnn.onnx").getFile());
        assertEquals(20,importGraph.graph().numVertices());
        val outputTanhOutput = importGraph.getFunctionForVertexId(15);
        assertNotNull(outputTanhOutput);
        assertTrue(outputTanhOutput instanceof Tanh);

        val pooling = importGraph.getFunctionForVertexId(16);
        assertTrue(pooling instanceof MaxPooling2D);

        val poolingCast = (MaxPooling2D) pooling;
        assertEquals(24,poolingCast.getConfig().getkH());
        assertEquals(24,poolingCast.getConfig().getkW());*/

    }

}
