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

package org.deeplearning4j.rl4j.network;

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.assertSame;
import static org.mockito.Mockito.*;

@RunWith(MockitoJUnitRunner.class)
public class QNetworkTest {

    private FeaturesLabels createFeaturesLabelsMock() {
        FeaturesLabels featuresLabelsMock = mock(FeaturesLabels.class);
        Features features = new Features(new INDArray[] { Nd4j.rand(1, 2) });
        when(featuresLabelsMock.getFeatures()).thenReturn(features);

        return featuresLabelsMock;
    }

    @Test
    public void when_callingCtorWithMLN_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        MultiLayerNetwork modelMock = mock(MultiLayerNetwork.class);
        FeaturesLabels featuresLabelsMock = createFeaturesLabelsMock();
        Gradient gradientMock = mock(Gradient.class);
        when(modelMock.gradient()).thenReturn(gradientMock);

        // Act
        QNetwork sut = buildQNetwork(modelMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.QValues);
        assertSame(gradientMock, results.getGradient(CommonGradientNames.QValues));
    }

    @Test
    public void when_callingCtorWithCG_expect_handlerUsesCorrectLabelAndGradientNames() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        FeaturesLabels featuresLabelsMock = createFeaturesLabelsMock();
        Gradient gradientMock = mock(Gradient.class);
        when(modelMock.gradient()).thenReturn(gradientMock);

        // Act
        QNetwork sut = buildQNetwork(modelMock);
        Gradients results = sut.computeGradients(featuresLabelsMock);

        // Assert
        verify(featuresLabelsMock, times(1)).getLabels(CommonLabelNames.QValues);
        assertSame(gradientMock, results.getGradient(CommonGradientNames.QValues));
    }

    @Test
    public void when_callingOutput_expect_resultHasCorrectNames() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        INDArray featuresData = Nd4j.rand(1, 2);
        Features features = new Features(new INDArray[] { featuresData });
        INDArray output = Nd4j.rand(1, 2);
        when(modelMock.output(featuresData)).thenReturn(new INDArray[] { output });

        // Act
        QNetwork sut = buildQNetwork(modelMock);
        NeuralNetOutput result = sut.output(features);

        // Assert
        assertSame(output, result.get(CommonOutputNames.QValues));
    }

    @Test
    public void when_callingClone_expect_clonedQNetwork() {
        // Arrange
        ComputationGraph modelMock = mock(ComputationGraph.class);
        when(modelMock.clone()).thenReturn(modelMock);

        // Act
        QNetwork sut = buildQNetwork(modelMock);
        QNetwork clone = sut.clone();

        // Assert
        assertNotSame(sut, clone);
        assertNotSame(sut.getNetworkHandler(), clone.getNetworkHandler());
    }

    private QNetwork buildQNetwork(ComputationGraph model) {
        return QNetwork.builder()
                .withNetwork(model)
                .build();
    }

    private QNetwork buildQNetwork(MultiLayerNetwork model) {
        return QNetwork.builder()
                .withNetwork(model)
                .build();
    }

}
