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

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.BertWordPiecePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.BertWordPieceTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.resources.Resources;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

@Slf4j
public class BertWordPieceTokenizerTests extends BaseDL4JTest {

    private File pathToVocab =  Resources.asFile("other/vocab.txt");
    private Charset c = StandardCharsets.UTF_8;

    public BertWordPieceTokenizerTests() throws IOException {
    }

    @Test
    public void testBertWordPieceTokenizer1() throws Exception {
        String toTokenize = "I saw a girl with a telescope.";
        TokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));
        int position = 1;
        while (tokenizer2.hasMoreTokens()) {
            String tok1 = tokenizer.nextToken();
            String tok2 = tokenizer2.nextToken();
            log.info("Position: [" + position + "], token1: '" + tok1 + "', token 2: '" + tok2 + "'");
            position++;
            assertEquals(tok1, tok2);

            String s2 = BertWordPiecePreProcessor.reconstructFromTokens(tokenizer.getTokens());
            assertEquals(toTokenize, s2);
        }
    }

    @Test
    public void testBertWordPieceTokenizer2() throws Exception {
        TokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);

        ClassPathResource resource = new ClassPathResource("reuters/5250");
        String str = FileUtils.readFileToString(resource.getFile());
        int stringCount = t.create(str).countTokens();
        int stringCount2 = t.create(resource.getInputStream()).countTokens();
        assertTrue(Math.abs(stringCount - stringCount2) < 2);
    }

    @Test
    public void testBertWordPieceTokenizer3() throws Exception {
        String toTokenize = "Donaudampfschifffahrtskapitänsmützeninnenfuttersaum";
        TokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));

        final List<String> expected = Arrays.asList("Donau", "##dam", "##pf", "##schiff", "##fahrt", "##skap", "##itä", "##ns", "##m", "##ützen", "##innen", "##fu", "##tter", "##sa", "##um");
        assertEquals(expected, tokenizer.getTokens());
        assertEquals(expected, tokenizer2.getTokens());

        String s2 = BertWordPiecePreProcessor.reconstructFromTokens(tokenizer.getTokens());
        assertEquals(toTokenize, s2);
    }

    @Test
    public void testBertWordPieceTokenizer4() throws Exception {
        String toTokenize = "I saw a girl with a telescope.";
        TokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));

        final List<String> expected = Arrays.asList("I", "saw", "a", "girl", "with", "a", "tele", "##scope", ".");
        assertEquals(expected, tokenizer.getTokens());
        assertEquals(expected, tokenizer2.getTokens());

        String s2 = BertWordPiecePreProcessor.reconstructFromTokens(tokenizer.getTokens());
        assertEquals(toTokenize, s2);
    }

    @Test
    @Ignore("AB 2019/05/24 - Disabled until dev branch merged - see issue #7657")
    public void testBertWordPieceTokenizer5() throws Exception {
        // Longest Token in Vocab is 22 chars long, so make sure splits on the edge are properly handled
        String toTokenize = "Donaudampfschifffahrts Kapitänsmützeninnenfuttersaum";
        TokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, false, false, c);
        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));

        final List<String> expected = Arrays.asList("Donau", "##dam", "##pf", "##schiff", "##fahrt", "##s", "Kapitän", "##sm", "##ützen", "##innen", "##fu", "##tter", "##sa", "##um");
        assertEquals(expected, tokenizer.getTokens());
        assertEquals(expected, tokenizer2.getTokens());

        String s2 = BertWordPiecePreProcessor.reconstructFromTokens(tokenizer.getTokens());
        assertEquals(toTokenize, s2);
    }

    @Test
    public void testBertWordPieceTokenizer6() throws Exception {
        String toTokenize = "I sAw A gIrL wItH a tElEsCoPe.";
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, true, true, c);

        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));

        final List<String> expected = Arrays.asList("i", "saw", "a", "girl", "with", "a", "tele", "##scope", ".");
        assertEquals(expected, tokenizer.getTokens());
        assertEquals(expected, tokenizer2.getTokens());

        String s2 = BertWordPiecePreProcessor.reconstructFromTokens(tokenizer.getTokens());
        assertEquals(toTokenize.toLowerCase(), s2);
    }

    @Test
    public void testBertWordPieceTokenizer7() throws Exception {
        String toTokenize = "I saw a girl with a telescope.";
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, true, true, c);

        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));

        final List<String> expected = Arrays.asList("i", "saw", "a", "girl", "with", "a", "tele", "##scope", ".");
        assertEquals(expected, tokenizer.getTokens());
        assertEquals(expected, tokenizer2.getTokens());

        String s2 = BertWordPiecePreProcessor.reconstructFromTokens(tokenizer.getTokens());
        assertEquals(toTokenize.toLowerCase(), s2);
    }

    @Test
    public void testBertWordPieceTokenizer8() throws Exception {
        //Insert some invalid characters...
        String toTokenize = "I saw a girl " + (char) 8 + " with a tele" + (char)7 + "scope.";
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, true, true, c);

        Tokenizer tokenizer = t.create(toTokenize);
        Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));

        final List<String> expected = Arrays.asList("i", "saw", "a", "girl", "with", "a", "tele", "##scope", ".");
        assertEquals(expected, tokenizer.getTokens());
        assertEquals(expected, tokenizer2.getTokens());
    }

    @Test
    public void testBertWordPieceTokenizer9() throws Exception {
        //Insert some invalid characters - without the preprocessing. This should fail

        for(String toTokenize : new String[]{
                "I saw a girl with a tele" + (char) 7 + "scope.",
                "I saw a girl " + (char) 8 + " with a tele" + (char)7 + "scope.",
                "\u23A0I saw a girl with a telescope.",
                "I saw a girl with a telescope.\u23A0"
        }) {
            BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(pathToVocab, true, true, c);
            t.setPreTokenizePreProcessor(null);

            try {
                t.create(toTokenize);
                fail("Expected exception: " + toTokenize);
            } catch (IllegalStateException e) {
                String m = e.getMessage();
                assertNotNull(m);
                m = m.toLowerCase();
                assertTrue(m, m.contains("invalid") && m.contains("token") && m.contains("preprocessor"));
            }

            try {
                t.create(new ByteArrayInputStream(toTokenize.getBytes()));
                fail("Expected exception: " + toTokenize);
            } catch (IllegalStateException e) {
                String m = e.getMessage();
                assertNotNull(m);
                m = m.toLowerCase();
                assertTrue(m, m.contains("invalid") && m.contains("token") && m.contains("preprocessor"));
            }
        }
    }


    @Test(timeout = 300000)
    public void testBertWordPieceTokenizer10() throws Exception {
        File f = Resources.asFile("deeplearning4j-nlp/bert/uncased_L-12_H-768_A-12/vocab.txt");
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(f, true, true, StandardCharsets.UTF_8);

        String s = "This is a sentence with Multiple Cases For Words. It should be coverted to Lower Case here.";

        Tokenizer tokenizer = t.create(s);
        List<String> list = tokenizer.getTokens();
        System.out.println(list);

        String s2 = BertWordPiecePreProcessor.reconstructFromTokens(list);
        String exp = s.toLowerCase();
        assertEquals(exp, s2);
    }

    @Test
    public void testTokenizerHandlesLargeContiguousWhitespace() throws Exception {
        StringBuilder sb = new StringBuilder();
        sb.append("apple.");
        for (int i = 0; i < 10000; i++) {
            sb.append(" ");
        }
        sb.append(".pen. .pineapple");

        File f = Resources.asFile("deeplearning4j-nlp/bert/uncased_L-12_H-768_A-12/vocab.txt");
        BertWordPieceTokenizerFactory t = new BertWordPieceTokenizerFactory(f, true, true, StandardCharsets.UTF_8);

        Tokenizer tokenizer = t.create(sb.toString());
        List<String> list = tokenizer.getTokens();
        System.out.println(list);

        assertEquals(8, list.size());
    }
}
