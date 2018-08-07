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

package org.nd4j.tensorflow.conversion;

import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.tensorflow;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.tensorflow.framework.GraphDef;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

public class TensorflowConversionTest {

    @Test
    public void testView() {
        INDArray matrix = Nd4j.linspace(1,8,8).reshape(2,4);
        INDArray view = matrix.slice(0);
        TensorflowConversion conversion =TensorflowConversion.getInstance();
        tensorflow.TF_Tensor tf_tensor = conversion.tensorFromNDArray(view);
        INDArray converted = conversion.ndArrayFromTensor(tf_tensor);
        assertEquals(view,converted);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNullArray() {
        INDArray array = Nd4j.create(2,2);
        array.setData(null);
        TensorflowConversion conversion =TensorflowConversion.getInstance();
        tensorflow.TF_Tensor tf_tensor = conversion.tensorFromNDArray(array);
        fail();
    }

    @Test
    public void testConversionFromNdArray() throws Exception {
        INDArray arr = Nd4j.linspace(1,4,4);
        TensorflowConversion tensorflowConversion =TensorflowConversion.getInstance();
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
        TensorflowConversion tensorflowConversion =TensorflowConversion.getInstance();
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());
        //byte[] content = Files.readAllBytes(Paths.get(new File("/home/agibsonccc/code/dl4j-test-resources/src/main/resources/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").toURI()));
        tensorflow.TF_Graph initializedGraphForNd4jDevices = tensorflowConversion.loadGraph(content);
        assertNotNull(initializedGraphForNd4jDevices);

        String deviceName = tensorflowConversion.defaultDeviceForThread();

        byte[] content2 = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());
        GraphDef graphDef1 = GraphDef.parseFrom(content2);
        for(int i = 0; i < graphDef1.getNodeCount(); i++)
            assertEquals(deviceName,graphDef1.getNode(i).getDevice());
        System.out.println(graphDef1);
    }




}
