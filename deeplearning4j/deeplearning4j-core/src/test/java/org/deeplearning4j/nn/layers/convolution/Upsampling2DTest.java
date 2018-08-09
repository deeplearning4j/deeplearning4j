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

package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Upsampling2D;
import org.deeplearning4j.nn.gradient.Gradient;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author Max Pumperla
 */
public class Upsampling2DTest extends BaseDL4JTest {

    private int nExamples = 1;
    private int depth = 20;
    private int nChannelsIn = 1;
    private int inputWidth = 28;
    private int inputHeight = 28;

    private int size = 2;
    private int outputWidth = inputWidth * size;
    private int outputHeight = inputHeight * size;

    private INDArray epsilon = Nd4j.ones(nExamples, depth, outputHeight, outputWidth);


    @Test
    public void testUpsampling() throws Exception {

        double[] outArray = new double[] {1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.};
        INDArray containedExpectedOut = Nd4j.create(outArray, new int[] {1, 1, 4, 4});
        INDArray containedInput = getContainedData();
        INDArray input = getData();
        Layer layer = getUpsamplingLayer();

        INDArray containedOutput = layer.activate(containedInput, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(containedExpectedOut.shape(), containedOutput.shape()));
        assertEquals(containedExpectedOut, containedOutput);

        INDArray output = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
        assertTrue(Arrays.equals(new long[] {nExamples, nChannelsIn, outputWidth, outputHeight},
                        output.shape()));
        assertEquals(nChannelsIn, output.size(1), 1e-4);
    }


    @Test
    public void testUpsampling2DBackprop() throws Exception {
        INDArray expectedContainedEpsilonInput =
                        Nd4j.create(new double[] {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.},
                                new int[] {1, 1, 4, 4});

        INDArray expectedContainedEpsilonResult = Nd4j.create(new double[] {4., 4., 4., 4.},
                        new int[] {1, 1, 2, 2});

        INDArray input = getContainedData();

        Layer layer = getUpsamplingLayer();
        layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        Pair<Gradient, INDArray> containedOutput = layer.backpropGradient(expectedContainedEpsilonInput, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(expectedContainedEpsilonResult, containedOutput.getSecond());
        assertEquals(null, containedOutput.getFirst().getGradientFor("W"));
        assertEquals(expectedContainedEpsilonResult.shape().length, containedOutput.getSecond().shape().length);

        INDArray input2 = getData();
        layer.activate(input2, false, LayerWorkspaceMgr.noWorkspaces());
        val depth = input2.size(1);

        epsilon = Nd4j.ones(5, depth, outputHeight, outputWidth);

        Pair<Gradient, INDArray> out = layer.backpropGradient(epsilon, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(input.shape().length, out.getSecond().shape().length);
        assertEquals(depth, out.getSecond().size(1));
    }


    private Layer getUpsamplingLayer() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).seed(123)
                        .layer(new Upsampling2D.Builder(size).build()).build();
        return conf.getLayer().instantiate(conf, null, 0, null, true);
    }

    public INDArray getData() throws Exception {
        DataSetIterator data = new MnistDataSetIterator(5, 5);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatures().reshape(nExamples, nChannelsIn, inputWidth, inputHeight);
    }

    private INDArray getContainedData() {
        INDArray ret = Nd4j.create
                (new double[] {1., 2., 3., 4.},
                        new int[] {1, 1, 2, 2});
        return ret;
    }

}
