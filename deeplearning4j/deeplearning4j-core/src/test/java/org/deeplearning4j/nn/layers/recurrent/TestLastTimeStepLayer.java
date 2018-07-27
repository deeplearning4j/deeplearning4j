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

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import static org.junit.Assert.assertEquals;

public class TestLastTimeStepLayer extends BaseDL4JTest {

    @Test
    public void testLastTimeStepVertex() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("lastTS", new LastTimeStep(new SimpleRnn.Builder()
                        .nIn(5).nOut(6).build()), "in")
                .setOutputs("lastTS")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        //First: test without input mask array
        Nd4j.getRandom().setSeed(12345);
        Layer l = graph.getLayer("lastTS");
        INDArray in = Nd4j.rand(new int[]{3, 5, 6});
        INDArray outUnderlying = ((LastTimeStepLayer)l).getUnderlying().activate(in, false, LayerWorkspaceMgr.noWorkspaces());
        INDArray expOut = outUnderlying.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(5));


        //Forward pass:
        INDArray outFwd = l.activate(in, false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expOut, outFwd);

        //Second: test with input mask array
        INDArray inMask = Nd4j.zeros(3, 6);
        inMask.putRow(0, Nd4j.create(new double[]{1, 1, 1, 0, 0, 0}));
        inMask.putRow(1, Nd4j.create(new double[]{1, 1, 1, 1, 0, 0}));
        inMask.putRow(2, Nd4j.create(new double[]{1, 1, 1, 1, 1, 0}));
        graph.setLayerMaskArrays(new INDArray[]{inMask}, null);

        expOut = Nd4j.zeros(3, 6);
        expOut.putRow(0, outUnderlying.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(2)));
        expOut.putRow(1, outUnderlying.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.point(3)));
        expOut.putRow(2, outUnderlying.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.point(4)));

        outFwd = l.activate(in, false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expOut, outFwd);

        TestUtils.testModelSerialization(graph);
    }

}
