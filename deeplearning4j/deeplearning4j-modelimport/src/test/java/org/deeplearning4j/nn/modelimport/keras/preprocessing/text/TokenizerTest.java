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

package org.deeplearning4j.nn.modelimport.keras.preprocessing.text;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Tests for Keras Tokenizer
 *
 * @author Max Pumperla
 */
public class TokenizerTest {

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
}
