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

package org.deeplearning4j.text.tokenization.tokenizer;

import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


/**
 * @author sonali
 */
public class NGramTokenizerTest {

    @Test
    public void testNGramTokenizer() throws Exception {
        String toTokenize = "Mary had a little lamb.";
        TokenizerFactory factory = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 1, 2);
        Tokenizer tokenizer = factory.create(toTokenize);
        Tokenizer tokenizer2 = factory.create(toTokenize);
        while (tokenizer.hasMoreTokens()) {
            assertEquals(tokenizer.nextToken(), tokenizer2.nextToken());
        }

        int stringCount = factory.create(toTokenize).countTokens();
        List<String> tokens = factory.create(toTokenize).getTokens();
        assertEquals(9, stringCount);

        assertTrue(tokens.contains("Mary"));
        assertTrue(tokens.contains("had"));
        assertTrue(tokens.contains("a"));
        assertTrue(tokens.contains("little"));
        assertTrue(tokens.contains("lamb."));
        assertTrue(tokens.contains("Mary had"));
        assertTrue(tokens.contains("had a"));
        assertTrue(tokens.contains("a little"));
        assertTrue(tokens.contains("little lamb."));

        factory = new NGramTokenizerFactory(new DefaultTokenizerFactory(), 2, 2);
        tokens = factory.create(toTokenize).getTokens();
        assertEquals(4, tokens.size());

        assertTrue(tokens.contains("Mary had"));
        assertTrue(tokens.contains("had a"));
        assertTrue(tokens.contains("a little"));
        assertTrue(tokens.contains("little lamb."));
    }
}
