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
package com.atilika.kuromoji.util;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class DictionaryEntryLineParserTest {

    private DictionaryEntryLineParser parser = new DictionaryEntryLineParser();

    @Test
    public void testTrivial() {
        assertArrayEquals(new String[] {"日本経済新聞", "日本 経済 新聞", "ニホン ケイザイ シンブン", "カスタム名詞"},
                        parser.parseLine("日本経済新聞,日本 経済 新聞,ニホン ケイザイ シンブン,カスタム名詞"));
    }

    @Test
    public void testQuotes() {
        assertArrayEquals(
                        new String[] {"Java Platform, Standard Edition", "Java Platform, Standard Edition",
                                        "Java Platform, Standard Edition", "カスタム名詞"},
                        parser.parseLine(
                                        "\"Java Platform, Standard Edition\",\"Java Platform, Standard Edition\",\"Java Platform, Standard Edition\",カスタム名詞"));
    }

    @Test
    public void testQuotedQuotes() {
        assertArrayEquals(new String[] {"Java \"Platform\"", "Java \"Platform\"", "Java \"Platform\"", "カスタム名詞"}, parser
                        .parseLine("\"Java \"\"Platform\"\"\",\"Java \"\"Platform\"\"\",\"Java \"\"Platform\"\"\",カスタム名詞"));
    }

    @Test
    public void testEmptyQuotedQuotes() {
        assertArrayEquals(new String[] {"\"", "\"", "quote", "punctuation"},
                        parser.parseLine("\"\"\"\",\"\"\"\",quote,punctuation"));
    }

    @Test
    public void testCSharp() {
        assertArrayEquals(new String[] {"C#", "C #", "シーシャープ", "プログラミング言語"},
                        parser.parseLine("\"C#\",\"C #\",シーシャープ,プログラミング言語"));
    }

    @Test
    public void testTab() {
        assertArrayEquals(new String[] {"A\tB", "A B", "A B", "tab"}, parser.parseLine("A\tB,A B,A B,tab"));
    }

    @Test
    public void testFrancoisWhiteBuffaloBota() {

        assertArrayEquals(
                        new String[] {"フランソワ\"ザホワイトバッファロー\"ボタ", "フランソワ\"ザホワイトバッファロー\"ボタ", "フランソワ\"ザホワイトバッファロー\"ボタ",
                                        "名詞"},
                        parser.parseLine(
                                        "\"フランソワ\"\"ザホワイトバッファロー\"\"ボタ\",\"フランソワ\"\"ザホワイトバッファロー\"\"ボタ\",\"フランソワ\"\"ザホワイトバッファロー\"\"ボタ\",名詞"));
    }

    @Test(expected = RuntimeException.class)
    public void testSingleQuote() {
        parser.parseLine("this is an entry with \"unmatched quote");
    }

    @Test(expected = RuntimeException.class)
    public void testUnmatchedQuote() {
        parser.parseLine("this is an entry with \"\"\"unmatched quote");
    }

    @Test
    public void testEscapeRoundTrip() {
        String original = "3,\"14";

        assertEquals("\"3,\"\"14\"", DictionaryEntryLineParser.escape(original));
        assertEquals(original, DictionaryEntryLineParser.unescape(DictionaryEntryLineParser.escape(original)));
    }

    @Test
    public void testUnescape() {
        assertEquals("A", DictionaryEntryLineParser.unescape("\"A\""));
        assertEquals("\"A\"", DictionaryEntryLineParser.unescape("\"\"\"A\"\"\""));

        assertEquals("\"", DictionaryEntryLineParser.unescape("\"\"\"\""));
        assertEquals("\"\"", DictionaryEntryLineParser.unescape("\"\"\"\"\"\""));
        assertEquals("\"\"\"", DictionaryEntryLineParser.unescape("\"\"\"\"\"\"\"\""));
        assertEquals("\"\"\"\"\"", DictionaryEntryLineParser.unescape("\"\"\"\"\"\"\"\"\"\"\"\""));
    }

    // TODO: these tests should be checked, right now they are documenting what is happening.
    @Test
    public void testParseInputString() throws Exception {
        String input = "日本経済新聞,1292,1292,4980,名詞,固有名詞,組織,*,*,*,日本経済新聞,ニホンケイザイシンブン,ニホンケイザイシンブン";
        String expected = Arrays.deepToString(new String[] {"日本経済新聞", "1292", "1292", "4980", "名詞", "固有名詞", "組織", "*",
                        "*", "*", "日本経済新聞", "ニホンケイザイシンブン", "ニホンケイザイシンブン"});
        assertEquals(expected, given(input));
    }

    @Test
    public void testParseInputStringWithQuotes() throws Exception {
        String input = "日本経済新聞,1292,1292,4980,名詞,固有名詞,組織,*,*,\"1,0\",日本経済新聞,ニホンケイザイシンブン,ニホンケイザイシンブン";
        String expected = Arrays.deepToString(new String[] {"日本経済新聞", "1292", "1292", "4980", "名詞", "固有名詞", "組織", "*",
                        "*", "1,0", "日本経済新聞", "ニホンケイザイシンブン", "ニホンケイザイシンブン"});
        assertEquals(expected, given(input));
    }

    @Test
    public void testQuoteEscape() throws Exception {
        String input = "日本経済新聞,1292,1292,4980,名詞,固有名詞,組織,*,*,\"1,0\",日本経済新聞,ニホンケイザイシンブン,ニホンケイザイシンブン";
        String expected = "\"日本経済新聞,1292,1292,4980,名詞,固有名詞,組織,*,*,\"\"1,0\"\",日本経済新聞,ニホンケイザイシンブン,ニホンケイザイシンブン\"";
        assertEquals(expected, parser.escape(input));
    }

    private String given(String input) {
        return Arrays.deepToString(parser.parseLine(input));
    }
}
