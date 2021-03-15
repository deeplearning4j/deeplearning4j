/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.onnxruntime.runner;

import org.junit.jupiter.api.Test;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;


public class OnnxRuntimeRunnerTests {



    @Test
    public void testAdd() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("add.onnx");
        File f = classPathResource.getFile();
        INDArray x = Nd4j.scalar(1.0f).reshape(1,1);
        INDArray y = Nd4j.scalar(1.0f).reshape(1,1);
        OnnxRuntimeRunner onnxRuntimeRunner = OnnxRuntimeRunner.builder()
                .modelUri(f.getAbsolutePath())
                .build();
        Map<String,INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x",x);
        inputs.put("y",y);
        Map<String, INDArray> exec = onnxRuntimeRunner.exec(inputs);
        INDArray z = exec.get("z");
        assertEquals(2.0,z.sumNumber().doubleValue(),1e-1);
    }

}
