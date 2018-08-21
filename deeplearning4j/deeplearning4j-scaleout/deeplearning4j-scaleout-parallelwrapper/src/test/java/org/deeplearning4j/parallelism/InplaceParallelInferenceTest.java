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

package org.deeplearning4j.parallelism;

import lombok.val;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.parallelism.inference.InferenceMode;
import org.junit.Test;

import static org.junit.Assert.*;

public class InplaceParallelInferenceTest {

    @Test
    public void testUpdateModel() {
        int nIn = 5;

        val conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("out0", new OutputLayer.Builder().nIn(nIn).nOut(4).build(), "in")
                .layer("out1", new OutputLayer.Builder().nIn(nIn).nOut(6).build(), "in")
                .setOutputs("out0", "out1")
                .build();

        val net = new ComputationGraph(conf);
        net.init();

        val pi = new ParallelInference.Builder(net)
                .inferenceMode(InferenceMode.INPLACE)
                .workers(2)
                .build();

        assertTrue(pi instanceof InplaceParallelInference);

        val models = pi.getCurrentModelsFromWorkers();

        assertEquals(2, models.length);

        for (val m:models) {
            assertNotNull(m);
            assertEquals(net.params(), m.params());
        }

        val conf2 = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .layer("out0", new OutputLayer.Builder().nIn(nIn).nOut(4).build(), "in")
                .layer("out1", new OutputLayer.Builder().nIn(nIn).nOut(6).build(), "in")
                .layer("out2", new OutputLayer.Builder().nIn(nIn).nOut(8).build(), "in")
                .setOutputs("out0", "out1", "out2")
                .build();

        val net2 = new ComputationGraph(conf2);
        net2.init();

        assertNotEquals(net.params(), net2.params());

        pi.updateModel(net2);

        val models2 = pi.getCurrentModelsFromWorkers();

        assertEquals(2, models2.length);

        for (val m:models2) {
            assertNotNull(m);
            assertEquals(net2.params(), m.params());
        }
    }
}