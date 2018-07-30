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

/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.atilika.kuromoji.ipadic;

import com.atilika.kuromoji.CommonCornerCasesTest;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static com.atilika.kuromoji.TestUtils.*;
import static org.junit.Assert.*;

public class TokenizerTest {

    private static Tokenizer tokenizer;

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {
        tokenizer = new Tokenizer();
    }

    @Test
    public void testSimpleSegmentation() {
        String input = "スペースステーションに行きます。うたがわしい。";
        String[] surfaces = {"スペース", "ステーション", "に", "行き", "ます", "。", "うたがわしい", "。"};
        List<Token> tokens = tokenizer.tokenize(input);
        assertTrue(tokens.size() == surfaces.length);
        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(surfaces[i], tokens.get(i).getSurface());
        }
    }

    @Test
    public void testSimpleReadings() {
        List<Token> tokens = tokenizer.tokenize("寿司が食べたいです。");
        assertTrue(tokens.size() == 6);
        assertEquals(tokens.get(0).getReading(), "スシ");
        assertEquals(tokens.get(1).getReading(), "ガ");
        assertEquals(tokens.get(2).getReading(), "タベ");
        assertEquals(tokens.get(3).getReading(), "タイ");
        assertEquals(tokens.get(4).getReading(), "デス");
        assertEquals(tokens.get(5).getReading(), "。");
    }

    @Test
    public void testSimpleReading() {
        List<Token> tokens = tokenizer.tokenize("郵税");
        assertEquals(tokens.get(0).getReading(), "ユウゼイ");
    }

    @Test
    public void testSimpleBaseFormKnownWord() {
        List<Token> tokens = tokenizer.tokenize("お寿司が食べたい。");
        assertTrue(tokens.size() == 6);
        assertEquals("食べ", tokens.get(3).getSurface());
        assertEquals("食べる", tokens.get(3).getBaseForm());

    }

    @Test
    public void testSimpleBaseFormUnknownWord() {
        List<Token> tokens = tokenizer.tokenize("アティリカ株式会社");
        assertTrue(tokens.size() == 2);
        assertFalse(tokens.get(0).isKnown());
        assertEquals("*", tokens.get(0).getBaseForm());
        assertTrue(tokens.get(1).isKnown());
        assertEquals("株式会社", tokens.get(1).getBaseForm());
    }

    @Test
    public void testYabottaiCornerCase() {
        List<Token> tokens = tokenizer.tokenize("やぼったい");
        assertEquals(1, tokens.size());
        assertEquals("やぼったい", tokens.get(0).getSurface());
    }

    @Test
    public void testTsukitoshaCornerCase() {
        List<Token> tokens = tokenizer.tokenize("突き通しゃ");
        assertEquals(1, tokens.size());
        assertEquals("突き通しゃ", tokens.get(0).getSurface());
    }

    @Test
    public void testIpadicTokenAPIs() throws Exception {
        List<Token> tokens = tokenizer.tokenize("お寿司が食べたい！");
        String[] pronunciations = {"オ", "スシ", "ガ", "タベ", "タイ", "！"};

        assertEquals(pronunciations.length, tokens.size());

        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(pronunciations[i], tokens.get(i).getPronunciation());
        }

        String[] conjugationForms = {"*", "*", "*", "連用形", "基本形", "*"};

        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(conjugationForms[i], tokens.get(i).getConjugationForm());
        }

        String[] conjugationTypes = {"*", "*", "*", "一段", "特殊・タイ", "*"};

        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(conjugationTypes[i], tokens.get(i).getConjugationType());
        }

        String[] posLevel1 = {"接頭詞", "名詞", "助詞", "動詞", "助動詞", "記号"};

        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(posLevel1[i], tokens.get(i).getPartOfSpeechLevel1());
        }

        String[] posLevel2 = {"名詞接続", "一般", "格助詞", "自立", "*", "一般"};

        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(posLevel2[i], tokens.get(i).getPartOfSpeechLevel2());
        }

        String[] posLevel3 = {"*", "*", "一般", "*", "*", "*"};

        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(posLevel3[i], tokens.get(i).getPartOfSpeechLevel3());
        }

        String[] posLevel4 = {"*", "*", "*", "*", "*", "*"};

        for (int i = 0; i < tokens.size(); i++) {
            assertEquals(posLevel4[i], tokens.get(i).getPartOfSpeechLevel4());
        }
    }

    @Test
    public void testCustomPenalties() {
        String input = "シニアソフトウェアエンジニアを探しています";

        Tokenizer customTokenizer = new Tokenizer.Builder().mode(Tokenizer.Mode.SEARCH).kanjiPenalty(3, 10000)
                        .otherPenalty(Integer.MAX_VALUE, 0).build();

        String[] expected1 = {"シニアソフトウェアエンジニア", "を", "探し", "て", "い", "ます"};

        assertTokenSurfacesEquals(Arrays.asList(expected1), customTokenizer.tokenize(input));

        Tokenizer searchTokenizer = new Tokenizer.Builder().mode(Tokenizer.Mode.SEARCH).build();

        String[] expected2 = {"シニア", "ソフトウェア", "エンジニア", "を", "探し", "て", "い", "ます"};

        assertTokenSurfacesEquals(Arrays.asList(expected2), searchTokenizer.tokenize(input));

    }

    @Test
    public void testNakaguroSplit() {
        Tokenizer defaultTokenizer = new Tokenizer();
        Tokenizer nakakuroSplittingTokenizer = new Tokenizer.Builder().isSplitOnNakaguro(true).build();

        String input = "ラレ・プールカリムの音楽が好き。";

        assertTokenSurfacesEquals(Arrays.asList("ラレ・プールカリム", "の", "音楽", "が", "好き", "。"),
                        defaultTokenizer.tokenize(input));
        assertTokenSurfacesEquals(Arrays.asList("ラレ", "・", "プールカリム", "の", "音楽", "が", "好き", "。"),
                        nakakuroSplittingTokenizer.tokenize(input));
    }

    @Test
    public void testAllFeatures() {
        Tokenizer tokenizer = new Tokenizer();
        String input = "寿司が食べたいです。";

        List<Token> tokens = tokenizer.tokenize(input);
        assertEquals("寿司\t名詞,一般,*,*,*,*,寿司,スシ,スシ", toString(tokens.get(0)));
        assertEquals("が\t助詞,格助詞,一般,*,*,*,が,ガ,ガ", toString(tokens.get(1)));
        assertEquals("食べ\t動詞,自立,*,*,一段,連用形,食べる,タベ,タベ", toString(tokens.get(2)));
        assertEquals("たい\t助動詞,*,*,*,特殊・タイ,基本形,たい,タイ,タイ", toString(tokens.get(3)));
        assertEquals("です\t助動詞,*,*,*,特殊・デス,基本形,です,デス,デス", toString(tokens.get(4)));
    }

    private String toString(Token token) {
        return token.getSurface() + "\t" + token.getAllFeatures();
    }

    @Test
    public void testCompactedTrieCrash() {
        String input = "＼ｍ";
        Tokenizer tokenizer = new Tokenizer();

        assertTokenSurfacesEquals(Arrays.asList("＼", "ｍ"), tokenizer.tokenize(input));
    }

    @Test
    public void testFeatureLengths() throws IOException {
        String userDictionary = "" + "gsf,gsf,ジーエスーエフ,カスタム名詞\n";

        Tokenizer tokenizer = new Tokenizer.Builder()
                        .userDictionary(new ByteArrayInputStream(userDictionary.getBytes(StandardCharsets.UTF_8)))
                        .build();

        assertEqualTokenFeatureLengths("ahgsfdajhgsfdこの丘はアクロポリスと呼ばれている。", tokenizer);
    }

    @Test
    public void testNewBocchan() throws IOException {
        assertTokenizedStreamEquals(new ClassPathResource("deeplearning4j-nlp-japanese/bocchan-ipadic-features.txt").getInputStream(),
                        new ClassPathResource("deeplearning4j-nlp-japanese/bocchan.txt").getInputStream(), tokenizer);
    }

    @Test
    public void testPunctuation() {
        CommonCornerCasesTest.testPunctuation(new Tokenizer());
    }
}
