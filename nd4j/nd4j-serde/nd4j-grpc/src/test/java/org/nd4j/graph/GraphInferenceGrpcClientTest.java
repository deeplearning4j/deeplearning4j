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

package org.nd4j.graph;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.execution.input.Operands;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.*;

@Slf4j
public class GraphInferenceGrpcClientTest {
    @Test
    public void testSimpleGraph_1() throws Exception {
        val exp = Nd4j.create(new double[] {-0.95938617, -1.20301781, 1.22260064, 0.50172403, 0.59972949, 0.78568028, 0.31609724, 1.51674747, 0.68013491, -0.05227458, 0.25903158,1.13243439}, new long[]{3, 1, 4});

        // configuring client
        val client = new GraphInferenceGrpcClient("127.0.0.1", 40123);

        val graphId = RandomUtils.nextLong(0, Long.MAX_VALUE);

        // preparing and registering graph (it's optional, and graph might be embedded into Docker image
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/expand_dim/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        client.registerGraph(graphId, tg, ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build());

        //defining input
        val input0 = Nd4j.create(new double[] {0.09753360, 0.76124972, 0.24693797, 0.13813169, 0.33144656, 0.08299957, 0.67197708, 0.80659380, 0.98274191, 0.63566073, 0.21592326, 0.54902743}, new int[] {3, 4});
        val operands = new Operands().addArgument("input_0", input0);

        // sending request and getting result
        val result = client.output(graphId, operands);
        assertEquals(exp, result.getById("output"));
    }

    @Test
    public void testSimpleGraph_2() throws Exception {
        val exp = Nd4j.create(new double[] {-0.95938617, -1.20301781, 1.22260064, 0.50172403, 0.59972949, 0.78568028, 0.31609724, 1.51674747, 0.68013491, -0.05227458, 0.25903158,1.13243439}, new long[]{3, 1, 4});

        // configuring client
        val client = new GraphInferenceGrpcClient("127.0.0.1", 40123);

        val graphId = RandomUtils.nextLong(0, Long.MAX_VALUE);

        // preparing and registering graph (it's optional, and graph might be embedded into Docker image
        val tg = TFGraphMapper.getInstance().importGraph(new ClassPathResource("tf_graphs/examples/expand_dim/frozen_model.pb").getInputStream());
        assertNotNull(tg);
        client.registerGraph(graphId, tg, ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).build());

        //defining input
        val input0 = Nd4j.create(new double[] {0.09753360, 0.76124972, 0.24693797, 0.13813169, 0.33144656, 0.08299957, 0.67197708, 0.80659380, 0.98274191, 0.63566073, 0.21592326, 0.54902743}, new int[] {3, 4});
        val operands = new Operands().addArgument(1, 0, input0);

        // sending request and getting result
        val result = client.output(graphId, operands);
        assertEquals(exp, result.getById("output"));
    }
}