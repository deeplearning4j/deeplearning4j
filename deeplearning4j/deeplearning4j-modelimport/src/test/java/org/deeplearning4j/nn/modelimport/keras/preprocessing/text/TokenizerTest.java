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

package org.deeplearning4j.nn.modelimport.keras.preprocessing.text;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for Keras Tokenizer
 *
 * @author Max Pumperla
 */
public class TokenizerTest extends BaseDL4JTest {

    @Test
    public void tokenizerBasics() {
        int numDocs = 5;
        int numWords = 12;

        KerasTokenizer tokenizer = new KerasTokenizer(numWords);

        String[] texts = new String[] {
                "Black then white are all I see",
                "In my infancy",
                "Red and yellow then came to be",
                "Reaching out to me",
                "Lets me see."
        };

        tokenizer.fitOnTexts(texts);
        assertEquals(numDocs, tokenizer.getDocumentCount().intValue());

        INDArray matrix = tokenizer.textsToMatrix(texts, TokenizerMode.BINARY);
        assertArrayEquals(new long[] {numDocs, numWords}, matrix.shape());

        Integer[][] sequences = tokenizer.textsToSequences(texts);

        tokenizer.sequencesToTexts(sequences);
        tokenizer.sequencesToMatrix(sequences, TokenizerMode.TFIDF);
        tokenizer.fitOnSequences(sequences);
    }

    @Test
    public void tokenizerParity(){
        // See #7448
        KerasTokenizer tokenize = new KerasTokenizer(1000);
        String[] itemsArray = new String[] { "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua." };
        tokenize.fitOnTexts(itemsArray);
        Map<String, Integer> index = tokenize.getWordIndex();
        Map<String, Integer> expectedIndex = new HashMap<>();
        expectedIndex.put("lorem", 1);
        expectedIndex.put("ipsum", 2);
        expectedIndex.put("dolor", 3);
        expectedIndex.put("sit", 4);
        expectedIndex.put("amet", 5);
        expectedIndex.put("consectetur", 6);
        expectedIndex.put("adipiscing", 7);
        expectedIndex.put("elit", 8);
        expectedIndex.put("sed", 9);
        expectedIndex.put("do", 10);
        expectedIndex.put("eiusmod", 11);
        expectedIndex.put("tempor", 12);
        expectedIndex.put("incididunt", 13);
        expectedIndex.put("ut", 14);
        expectedIndex.put("labore", 15);
        expectedIndex.put("et", 16);
        expectedIndex.put("dolore", 17);
        expectedIndex.put("magna", 18);
        expectedIndex.put("aliqua", 19);
        assertEquals(expectedIndex.size(), index.size());
        for (Map.Entry<String, Integer> entry: expectedIndex.entrySet()){
            assertEquals(entry.getValue(), index.get(entry.getKey()));
        }
        
    }
}
