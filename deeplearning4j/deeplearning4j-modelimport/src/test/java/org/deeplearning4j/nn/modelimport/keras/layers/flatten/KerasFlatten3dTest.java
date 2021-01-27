/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.modelimport.keras.layers.flatten;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.Cnn3DToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.PreprocessorVertex;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.junit.Test;
import org.nd4j.common.io.ClassPathResource;

import java.io.InputStream;

import static org.junit.Assert.*;

public class KerasFlatten3dTest {


    @Test
    public void testFlatten3d() throws Exception {
        ClassPathResource classPathResource = new ClassPathResource("modelimport/keras/weights/flatten_3d.hdf5");
        try(InputStream inputStream = classPathResource.getInputStream()) {
            ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights(inputStream);
            assertNotNull(computationGraph);
            assertEquals(3,computationGraph.getVertices().length);
            GraphVertex[] vertices = computationGraph.getVertices();
            assertTrue(vertices[1] instanceof PreprocessorVertex);
            PreprocessorVertex preprocessorVertex = (PreprocessorVertex) vertices[1];
            InputPreProcessor preProcessor = preprocessorVertex.getPreProcessor();
            assertTrue(preProcessor instanceof Cnn3DToFeedForwardPreProcessor);
            Cnn3DToFeedForwardPreProcessor cnn3DToFeedForwardPreProcessor = (Cnn3DToFeedForwardPreProcessor) preProcessor;
            assertTrue(cnn3DToFeedForwardPreProcessor.isNCDHW());
            assertEquals(10,cnn3DToFeedForwardPreProcessor.getInputDepth());
            assertEquals(10,cnn3DToFeedForwardPreProcessor.getInputHeight());
            assertEquals(1,cnn3DToFeedForwardPreProcessor.getNumChannels());
            assertEquals(10,cnn3DToFeedForwardPreProcessor.getInputWidth());
            System.out.println(cnn3DToFeedForwardPreProcessor);
        }
    }

}
