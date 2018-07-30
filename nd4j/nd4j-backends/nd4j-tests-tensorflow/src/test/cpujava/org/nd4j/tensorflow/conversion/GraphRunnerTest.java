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

import com.github.os72.protobuf351.util.JsonFormat;
import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.ConfigProto;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class GraphRunnerTest {

    @Test
    public void testGraphRunner() throws Exception {
        List<String> inputs = Arrays.asList("input_0","input_1");
        byte[] content = IOUtils.toByteArray(new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getInputStream());

        try(GraphRunner graphRunner = new GraphRunner(content,inputs)) {
            runGraphRunnerTest(graphRunner);
        }
    }

    @Test
    public void testGraphRunnerFilePath() throws Exception {
        List<String> inputs = Arrays.asList("input_0","input_1");
        File file = new ClassPathResource("/tf_graphs/nd4j_convert/simple_graph/frozen_model.pb").getFile();
        try(GraphRunner graphRunner = new GraphRunner(file.getAbsolutePath(),inputs)) {
            runGraphRunnerTest(graphRunner);
        }
    }

    @Test
    public void testInputOutputResolution() throws Exception {
        ClassPathResource lenetPb = new ClassPathResource("tf_graphs/lenet_frozen.pb");
        byte[] content = IOUtils.toByteArray(lenetPb.getInputStream());
        GraphRunner graphRunner = new GraphRunner(content,Arrays.asList("Reshape/tensor"));
        assertEquals(1,graphRunner.getInputOrder().size());
        assertEquals(1,graphRunner.getOutputOrder().size());
    }

    private void runGraphRunnerTest(GraphRunner graphRunner) throws Exception {

        org.tensorflow.framework.ConfigProto.Builder builder = org.tensorflow.framework.ConfigProto.newBuilder();
        String json = graphRunner.sessionOptionsToJson();
        JsonFormat.parser().merge(json,builder);
        org.tensorflow.framework.ConfigProto build = builder.build();
        assertEquals(build,graphRunner.getProtoBufConfigProto());
        assertNotNull(graphRunner.getInputOrder());
        assertNotNull(graphRunner.getOutputOrder());


        org.tensorflow.framework.ConfigProto configProto1 = GraphRunner.fromJson(json);

        assertEquals(graphRunner.getProtoBufConfigProto(),configProto1);
        assertEquals(2,graphRunner.getInputOrder().size());
        assertEquals(1,graphRunner.getOutputOrder().size());

        INDArray input1 = Nd4j.linspace(1,4,4).reshape(4);
        INDArray input2 = Nd4j.linspace(1,4,4).reshape(4);

        Map<String,INDArray> inputs = new LinkedHashMap<>();
        inputs.put("input_0",input1);
        inputs.put("input_1",input2);

        for(int i = 0; i < 2; i++) {
            Map<String,INDArray> outputs = graphRunner.run(inputs);

            INDArray assertion = input1.add(input2);
            assertEquals(assertion,outputs.get("output"));
        }

    }



}
