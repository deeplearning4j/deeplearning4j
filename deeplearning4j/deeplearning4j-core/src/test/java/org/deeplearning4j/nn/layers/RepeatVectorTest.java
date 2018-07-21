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

package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.misc.RepeatVector;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class RepeatVectorTest {

    private int REPEAT = 4;


    private Layer getRepeatVectorLayer() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
                .layer(new RepeatVector.Builder(REPEAT).build()).build();
        return conf.getLayer().instantiate(conf, null, 0,
                null, false);
    }

    @Test
    public void testRepeatVector() {

        double[] arr = new double[] {1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3.};
        INDArray expectedOut = Nd4j.create(arr, new long[] {1, 3, REPEAT}, 'f');
        INDArray input = Nd4j.create(new double[] {1., 2., 3.}, new long[] {1, 3});
        Layer layer = getRepeatVectorLayer();

        INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(expectedOut.shape(), output.shape()));
        assertEquals(expectedOut, output);

        INDArray epsilon = Nd4j.ones(1,3,4);

        Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
        INDArray outEpsilon = out.getSecond();
        INDArray expectedEpsilon = Nd4j.create(new double[] {4., 4., 4.}, new long[] {1, 3});
        assertEquals(expectedEpsilon, outEpsilon);
    }
}
