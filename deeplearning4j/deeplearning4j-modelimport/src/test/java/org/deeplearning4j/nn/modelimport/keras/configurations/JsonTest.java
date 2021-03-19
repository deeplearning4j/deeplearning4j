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
package org.deeplearning4j.nn.modelimport.keras.configurations;

import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.KerasFlattenRnnPreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.PermutePreprocessor;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.ReshapePreprocessor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Json Test")
class JsonTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Json Preprocessors")
    void testJsonPreprocessors() throws Exception {
        InputPreProcessor[] pp = new InputPreProcessor[] { new KerasFlattenRnnPreprocessor(10, 5), new PermutePreprocessor(new int[] { 0, 1, 2 }), new ReshapePreprocessor(new long[] { 10, 10 }, new long[] { 100, 1 }, true, null) };
        for (InputPreProcessor p : pp) {
            String s = NeuralNetConfiguration.mapper().writeValueAsString(p);
            InputPreProcessor p2 = NeuralNetConfiguration.mapper().readValue(s, InputPreProcessor.class);
            assertEquals(p, p2);
        }
    }
}
