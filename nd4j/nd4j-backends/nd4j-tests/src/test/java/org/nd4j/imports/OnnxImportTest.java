/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.imports;

import lombok.val;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;

import static org.junit.Assert.assertArrayEquals;

@RunWith(Parameterized.class)
public class OnnxImportTest extends BaseNd4jTest {


    public OnnxImportTest(Nd4jBackend backend) {
        super(backend);
    }


    @Override
    public char ordering() {
        return 'c';
    }


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
