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
}
