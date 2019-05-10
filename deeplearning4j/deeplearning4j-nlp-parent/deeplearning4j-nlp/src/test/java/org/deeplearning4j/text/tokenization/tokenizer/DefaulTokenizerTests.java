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

package org.deeplearning4j.text.tokenization.tokenizer;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.ByteArrayInputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DefaulTokenizerTests {

    protected static final Logger log = LoggerFactory.getLogger(DefaulTokenizerTests.class);

    @Test
    public void testDefaultTokenizer1() throws Exception {
        String toTokenize = "Mary had a little lamb.";
        TokenizerFactory t = new DefaultTokenizerFactory();
        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));
        int position = 1;
        while (tokenizer2.hasMoreTokens()) {
            String tok1 = tokenizer.nextToken();
            String tok2 = tokenizer2.nextToken();
            log.info("Position: [" + position + "], token1: '" + tok1 + "', token 2: '" + tok2 + "'");
            position++;
            assertEquals(tok1, tok2);
        }


        ClassPathResource resource = new ClassPathResource("reuters/5250");
        String str = FileUtils.readFileToString(resource.getFile());
        int stringCount = t.create(str).countTokens();
        int stringCount2 = t.create(resource.getInputStream()).countTokens();
        assertTrue(Math.abs(stringCount - stringCount2) < 2);
    }

    @Test
    public void testDefaultTokenizer2() throws Exception {
        String toTokenize = "Mary had a little lamb.";
        TokenizerFactory t = new DefaultTokenizerFactory();
        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));
        tokenizer2.countTokens();
        while (tokenizer.hasMoreTokens()) {
            String tok1 = tokenizer.nextToken();
            String tok2 = tokenizer2.nextToken();
            assertEquals(tok1, tok2);
        }


        System.out.println("-----------------------------------------------");

        ClassPathResource resource = new ClassPathResource("reuters/5250");
        String str = FileUtils.readFileToString(resource.getFile());
        int stringCount = t.create(str).countTokens();
        int stringCount2 = t.create(resource.getInputStream()).countTokens();

        log.info("String tok: [" + stringCount + "], Stream tok: [" + stringCount2 + "], Difference: "
                        + Math.abs(stringCount - stringCount2));

        assertTrue(Math.abs(stringCount - stringCount2) < 2);
    }

    @Test
    public void testDefaultTokenizer3() throws Exception {
        String toTokenize = "Mary had a little lamb.";
        TokenizerFactory t = new DefaultTokenizerFactory();
        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));
        int position = 1;
        while (tokenizer2.hasMoreTokens()) {
            String tok1 = tokenizer.nextToken();
            String tok2 = tokenizer2.nextToken();
            log.info("Position: [" + position + "], token1: '" + tok1 + "', token 2: '" + tok2 + "'");
            position++;
            assertEquals(tok1, tok2);
        }
    }

    @Test
    public void testDefaultStreamTokenizer() throws Exception {
        String toTokenize = "Mary had a little lamb.";
        TokenizerFactory t = new DefaultTokenizerFactory();
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));

        assertEquals(5, tokenizer2.countTokens());

        int cnt = 0;
        while (tokenizer2.hasMoreTokens()) {
            String tok1 = tokenizer2.nextToken();
            log.info(tok1);
            cnt++;
        }

        assertEquals(5, cnt);
    }


}
