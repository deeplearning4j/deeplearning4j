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
package org.nd4j.tensorflowlite.runner;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Tag(TagNames.FILE_IO)
@Tag(TagNames.DL4J_OLD_API)
@NativeTag
public class TensorFlowLiteRunnerTests {

    @Test
    public void testAdd() throws Exception {
        if(!Nd4j.getBackend().getEnvironment().isCPU())
            return;
        ClassPathResource classPathResource = new ClassPathResource("add.bin");
        File f = classPathResource.getFile();
        INDArray input = Nd4j.createFromArray(1.0f, 2.0f, 3.0f).reshape(1,1,1,3).broadcast(1,8,8,3);
        TensorFlowLiteRunner tensorFlowLiteRunner = TensorFlowLiteRunner.builder()
                .modelUri(f.getAbsolutePath())
                .build();
        Map<String,INDArray> inputs = new LinkedHashMap<>();
        inputs.put("input",input);
        Map<String, INDArray> exec = tensorFlowLiteRunner.exec(inputs);
        INDArray output = exec.get("output");
        assertEquals(3.0,output.getDouble(0),1e-1);
        assertEquals(6.0,output.getDouble(1),1e-1);
        assertEquals(9.0,output.getDouble(2),1e-1);
    }

}
