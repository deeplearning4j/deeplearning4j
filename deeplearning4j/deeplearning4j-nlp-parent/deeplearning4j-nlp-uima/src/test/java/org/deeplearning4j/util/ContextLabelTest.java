/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.util;

import org.deeplearning4j.text.movingwindow.ContextLabelRetriever;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Basic test case for the context label test
 */
public class ContextLabelTest {
    private static final Logger log = LoggerFactory.getLogger(ContextLabelTest.class);
    private TokenizerFactory tokenizerFactory;

    @Before
    public void init() throws Exception {
        if (tokenizerFactory == null) {
            tokenizerFactory = new UimaTokenizerFactory(false);
        }
    }

    @Test
    public void testBasicLabel() {
        String labeledSentence = "<NEGATIVE> This sucks really bad </NEGATIVE> .";
        Pair<String, org.nd4j.linalg.collection.MultiDimensionalMap<Integer, Integer, String>> ret =
                        ContextLabelRetriever.stringWithLabels(labeledSentence, tokenizerFactory);
        //positive and none
        assertEquals(2, ret.getSecond().size());
        List<String> vals = new ArrayList<>(ret.getSecond().values());
        assertEquals(true, vals.contains("NEGATIVE"));
        assertEquals(true, vals.contains("none"));
        assertEquals("This sucks really bad .", ret.getFirst());
    }


}
