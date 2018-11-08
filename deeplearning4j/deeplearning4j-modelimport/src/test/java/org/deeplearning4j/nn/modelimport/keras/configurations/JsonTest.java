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

package org.deeplearning4j.nn.modelimport.keras.configurations;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.KerasFlattenRnnPreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.PermutePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class JsonTest {

    @Test
    public void testJsonPreprocessors() throws Exception {
        InputPreProcessor[] pp = new InputPreProcessor[] {
                new KerasFlattenRnnPreprocessor(10, 5),
                new PermutePreprocessor(new int[]{0,1,2}),
                new ReshapePreprocessor(new long[]{10,10}, new long[]{100,1}),
                new TensorFlowCnnToFeedForwardPreProcessor()

        };
        for(InputPreProcessor p : pp ){
            String s = NeuralNetConfiguration.mapper().writeValueAsString(p);
            InputPreProcessor p2 = NeuralNetConfiguration.mapper().readValue(s, InputPreProcessor.class);
            assertEquals(p, p2);
        }

    }
}
