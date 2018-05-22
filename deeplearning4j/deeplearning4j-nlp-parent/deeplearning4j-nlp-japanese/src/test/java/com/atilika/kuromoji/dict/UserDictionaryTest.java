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
package com.atilika.kuromoji.dict;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class UserDictionaryTest {

    @Test
    public void testLookup() throws IOException {
        UserDictionary dictionary = new UserDictionary(getResource("userdict.txt"), 9, 7, 0);

        List<UserDictionary.UserDictionaryMatch> matches = dictionary.findUserDictionaryMatches("関西国際空港に行った");

        // Length should be three 関西, 国際, 空港
        assertEquals(3, matches.size());

        // Test positions
        assertEquals(0, matches.get(0).getMatchStartIndex()); // index of 関西
        assertEquals(2, matches.get(1).getMatchStartIndex()); // index of 国際
        assertEquals(4, matches.get(2).getMatchStartIndex()); // index of 空港

        // Test lengths
        assertEquals(2, matches.get(0).getMatchLength()); // length of 関西
        assertEquals(2, matches.get(1).getMatchLength()); // length of 国際
        assertEquals(2, matches.get(2).getMatchLength()); // length of 空港

        List<UserDictionary.UserDictionaryMatch> matches2 = dictionary.findUserDictionaryMatches("関西国際空港と関西国際空港に行った");
        assertEquals(6, matches2.size());
    }

    @Test
    public void testIpadicFeatures() throws IOException {
        UserDictionary dictionary = new UserDictionary(getResource("userdict.txt"), 9, 7, 0);

        assertEquals("カスタム名詞,*,*,*,*,*,*,ニホン,*", dictionary.getAllFeatures(100000000));
    }

    @Test
    public void testJumanDicFeatures() throws IOException {
        UserDictionary dictionary = new UserDictionary(getResource("userdict.txt"), 7, 5, 0);

        assertEquals("カスタム名詞,*,*,*,*,ニホン,*", dictionary.getAllFeatures(100000000));
    }

    @Test
    public void testNaistJDicFeatures() throws IOException {
        UserDictionary dictionary = new UserDictionary(getResource("userdict.txt"), 11, 7, 0);
        // This is a sample naist-jdic entry:
        //
        //   葦登,1358,1358,4975,名詞,一般,*,*,*,*,葦登,ヨシノボリ,ヨシノボリ,,
        //
        // How should we treat the last features in the user dictionary?  They seem empty, but we return * for them...
        assertEquals("カスタム名詞,*,*,*,*,*,*,ニホン,*,*,*", dictionary.getAllFeatures(100000000));
    }

    @Test
    public void testUniDicFeatures() throws IOException {
        UserDictionary dictionary = new UserDictionary(getResource("userdict.txt"), 13, 7, 0);

        assertEquals("カスタム名詞,*,*,*,*,*,*,ニホン,*,*,*,*,*", dictionary.getAllFeatures(100000000));
    }

    @Test
    public void testUniDicExtendedFeatures() throws IOException {
        UserDictionary dictionary = new UserDictionary(getResource("userdict.txt"), 22, 13, 0);

        assertEquals("カスタム名詞,*,*,*,*,*,*,*,*,*,*,*,*,ニホン,*,*,*,*,*,*,*,*", dictionary.getAllFeatures(100000000));
    }

    @Test
    public void testUserDictionaryEntries() throws IOException {
        String userDictionaryEntry = "クロ,クロ,クロ,カスタム名詞";
        UserDictionary dictionary = new UserDictionary(
                        new ByteArrayInputStream(userDictionaryEntry.getBytes(StandardCharsets.UTF_8)), 9, 7, 0);
        List<UserDictionary.UserDictionaryMatch> matches = dictionary.findUserDictionaryMatches("この丘はアクロポリスと呼ばれている");
        assertEquals(1, matches.size());
        assertEquals(5, matches.get(0).getMatchStartIndex());
    }

    @Test
    public void testOverlappingUserDictionaryEntries() throws IOException {
        String userDictionaryEntries = "" + "クロ,クロ,クロ,カスタム名詞\n" + "アクロ,アクロ,アクロ,カスタム名詞";
        UserDictionary dictionary = new UserDictionary(
                        new ByteArrayInputStream(userDictionaryEntries.getBytes(StandardCharsets.UTF_8)), 9, 7, 0);
        List<UserDictionary.UserDictionaryMatch> positions = dictionary.findUserDictionaryMatches("この丘はアクロポリスと呼ばれている");
        assertEquals(4, positions.get(0).getMatchStartIndex());
        assertEquals(2, positions.size());
    }

    private InputStream getResource(String resource) {
        return this.getClass().getClassLoader().getResourceAsStream(resource);
    }
}
