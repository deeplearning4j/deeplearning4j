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

package org.deeplearning4j.util;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.text.movingwindow.ContextLabelRetriever;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.nlp.uima.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.common.collection.MultiDimensionalMap;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Basic test case for the context label test
 */
public class ContextLabelTest extends BaseDL4JTest {
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
        Pair<String, MultiDimensionalMap<Integer, Integer, String>> ret =
                        ContextLabelRetriever.stringWithLabels(labeledSentence, tokenizerFactory);
        //positive and none
        assertEquals(2, ret.getSecond().size());
        List<String> vals = new ArrayList<>(ret.getSecond().values());
        assertEquals(true, vals.contains("NEGATIVE"));
        assertEquals(true, vals.contains("none"));
        assertEquals("This sucks really bad .", ret.getFirst());
    }


}
