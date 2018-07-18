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
package com.atilika.kuromoji.trie;

import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class PatriciaTrieTest {

    @Test
    public void testRomaji() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.put("a", "a");
        trie.put("b", "b");
        trie.put("ab", "ab");
        trie.put("bac", "bac");
        assertEquals("a", trie.get("a"));
        assertEquals("bac", trie.get("bac"));
        assertEquals("b", trie.get("b"));
        assertEquals("ab", trie.get("ab"));
        assertNull(trie.get("nonexistant"));
    }

    @Test
    public void testJapanese() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.put("寿司", "sushi");
        trie.put("刺身", "sashimi");
        assertEquals("sushi", trie.get("寿司"));
        assertEquals("sashimi", trie.get("刺身"));
    }

    @Test(expected = NullPointerException.class)
    public void testNull() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.put("null", null);
        assertEquals(null, trie.get("null"));
        trie.put(null, "null"); // Throws NullPointerException
        assertTrue(false);
    }

    @Test
    public void testRandom() {
        // Generate random strings
        List<String> randoms = new ArrayList<>();
        for (int i = 0; i < 100000; i++) {
            randoms.add(UUID.randomUUID().toString());
        }
        // Insert them
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        for (String random : randoms) {
            trie.put(random, random);
        }
        // Get and test them
        for (String random : randoms) {
            assertEquals(random, trie.get(random));
            assertTrue(trie.containsKey(random));
        }
    }

    @Test
    public void testPutAll() {
        // Generate random strings
        Map<String, String> randoms = new HashMap<>();
        for (int i = 0; i < 10000; i++) {
            String random = UUID.randomUUID().toString();
            randoms.put(random, random);
        }
        // Insert them
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.putAll(randoms);

        // Get and test them
        for (Map.Entry<String, String> random : randoms.entrySet()) {
            assertEquals(random.getValue(), trie.get(random.getKey()));
            assertTrue(trie.containsKey(random.getKey()));
        }
    }

    @Test
    public void testLongString() {
        String longMovieTitle = "マルキ・ド・サドの演出のもとにシャラントン精神病院患者たちによって演じられたジャン＝ポール・マラーの迫害と暗殺";

        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.put(longMovieTitle, "found it");

        assertEquals("found it", trie.get(longMovieTitle));
    }

    @Test(expected = ClassCastException.class)
    public void testUnsupportedType() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.put("hello", "world");
        assertTrue(trie.containsKey("hello"));
        trie.containsKey(new Integer(1));
        assertTrue(false);
    }

    @Test
    public void testEmpty() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        assertTrue(trie.isEmpty());
        trie.put("hello", "world");
        assertFalse(trie.isEmpty());
    }

    @Test
    public void testEmptyInsert() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        assertTrue(trie.isEmpty());
        trie.put("", "i am empty bottle of beer!");
        assertFalse(trie.isEmpty());
        assertEquals("i am empty bottle of beer!", trie.get(""));
        trie.put("", "...and i'm an empty bottle of sake");
        assertEquals("...and i'm an empty bottle of sake", trie.get(""));
    }

    @Test
    public void testClear() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        assertTrue(trie.isEmpty());
        assertEquals(0, trie.size());
        trie.put("hello", "world");
        trie.put("world", "hello");
        assertFalse(trie.isEmpty());
        trie.clear();
        assertTrue(trie.isEmpty());
        assertEquals(0, trie.size());
    }

    @Test
    public void testNaiveCollections() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.put("寿司", "sushi");
        trie.put("刺身", "sashimi");
        trie.put("そば", "soba");
        trie.put("ラーメン", "ramen");
        // Test keys
        assertEquals(4, trie.keySet().size());
        assertTrue(trie.keySet().containsAll(Arrays.asList(new String[] {"寿司", "そば", "ラーメン", "刺身"})));
        // Test values
        assertEquals(4, trie.values().size());
        assertTrue(trie.values().containsAll(Arrays.asList(new String[] {"sushi", "soba", "ramen", "sashimi"})));
    }

    @Test
    public void testEscapeChars() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        trie.put("new", "no error");
        assertFalse(trie.containsKeyPrefix("new\na"));
        assertFalse(trie.containsKeyPrefix("\n"));
        assertFalse(trie.containsKeyPrefix("\t"));
    }

    @Test
    public void testPrefix() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        String[] tokyoPlaces = new String[] {"Hachiōji", "Tachikawa", "Musashino", "Mitaka", "Ōme", "Fuchū", "Akishima",
                        "Chōfu", "Machida", "Koganei", "Kodaira", "Hino", "Higashimurayama", "Kokubunji", "Kunitachi",
                        "Fussa", "Komae", "Higashiyamato", "Kiyose", "Higashikurume", "Musashimurayama", "Tama",
                        "Inagi", "Hamura", "Akiruno", "Nishitōkyō"};
        for (int i = 0; i < tokyoPlaces.length; i++) {
            trie.put(tokyoPlaces[i], tokyoPlaces[i]);
        }

        // Prefixes of Kodaira
        assertTrue(trie.containsKeyPrefix("K"));
        assertTrue(trie.containsKeyPrefix("Ko"));
        assertTrue(trie.containsKeyPrefix("Kod"));
        assertTrue(trie.containsKeyPrefix("Koda"));
        assertTrue(trie.containsKeyPrefix("Kodai"));
        assertTrue(trie.containsKeyPrefix("Kodair"));
        assertTrue(trie.containsKeyPrefix("Kodaira"));
        assertFalse(trie.containsKeyPrefix("Kodaira "));
        assertFalse(trie.containsKeyPrefix("Kodaira  "));
        assertTrue(trie.get("Kodaira") != null);

        // Prefixes of Fussa
        assertFalse(trie.containsKeyPrefix("fu"));
        assertTrue(trie.containsKeyPrefix("Fu"));
        assertTrue(trie.containsKeyPrefix("Fus"));
    }

    @Test
    public void testTextScan() {
        PatriciaTrie<String> trie = new PatriciaTrie<>();
        String[] terms = new String[] {"お寿司", "sushi", "美味しい", "tasty", "日本", "japan", "だと思います", "i think", "料理",
                        "food", "日本料理", "japanese food", "一番", "first and foremost",};
        for (int i = 0; i < terms.length; i += 2) {
            trie.put(terms[i], terms[i + 1]);
        }

        String text = "日本料理の中で、一番美味しいのはお寿司だと思います。すぐ日本に帰りたいです。";
        StringBuilder builder = new StringBuilder();

        int startIndex = 0;
        while (startIndex < text.length()) {
            int matchLength = 0;
            while (trie.containsKeyPrefix(text.substring(startIndex, startIndex + matchLength + 1))) {
                matchLength++;
            }
            if (matchLength > 0) {
                String match = text.substring(startIndex, startIndex + matchLength);
                builder.append("[");
                builder.append(match);
                builder.append("|");
                builder.append(trie.get(match));
                builder.append("]");
                startIndex += matchLength;
            } else {
                builder.append(text.charAt(startIndex));
                startIndex++;
            }
        }
        assertEquals("[日本料理|japanese food]の中で、[一番|first and foremost][美味しい|tasty]のは[お寿司|sushi][だと思います|i think]。すぐ[日本|japan]に帰りたいです。",
                        builder.toString());
    }

    @Test
    public void testMultiThreadedTrie() throws InterruptedException {
        final int numThreads = 10;
        final int perThreadRuns = 500000;
        final int keySetSize = 1000;

        final List<Thread> threads = new ArrayList<>();
        final List<String> randoms = new ArrayList<>();

        final PatriciaTrie<Integer> trie = new PatriciaTrie<>();

        for (int i = 0; i < keySetSize; i++) {
            String random = UUID.randomUUID().toString();
            randoms.add(random);
            trie.put(random, i);
        }

        for (int i = 0; i < numThreads; i++) {
            Thread thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int run = 0; run < perThreadRuns; run++) {
                        int randomIndex = (int) (Math.random() * randoms.size());
                        String random = randoms.get(randomIndex);

                        // Test retrieve
                        assertEquals(randomIndex, (int) trie.get(random));

                        int randomPrefixLength = (int) (Math.random() * random.length());

                        // Test random prefix length prefix match
                        assertTrue(trie.containsKeyPrefix(random.substring(0, randomPrefixLength)));
                    }
                }
            });
            threads.add(thread);
            thread.start();
        }

        for (Thread thread : threads) {
            thread.join();
        }

        assertTrue(true);
    }

    @Test
    public void testSimpleKey() {
        PatriciaTrie.KeyMapper<String> keyMapper = new PatriciaTrie.StringKeyMapper();
        String key = "abc";

        // a = U+0061 = 0000 0000 0110 0001
        assertFalse(keyMapper.isSet(0, key));
        assertFalse(keyMapper.isSet(1, key));
        assertFalse(keyMapper.isSet(2, key));
        assertFalse(keyMapper.isSet(3, key));

        assertFalse(keyMapper.isSet(4, key));
        assertFalse(keyMapper.isSet(5, key));
        assertFalse(keyMapper.isSet(6, key));
        assertFalse(keyMapper.isSet(7, key));

        assertFalse(keyMapper.isSet(8, key));
        assertTrue(keyMapper.isSet(9, key));
        assertTrue(keyMapper.isSet(10, key));
        assertFalse(keyMapper.isSet(11, key));

        assertFalse(keyMapper.isSet(12, key));
        assertFalse(keyMapper.isSet(13, key));
        assertFalse(keyMapper.isSet(14, key));
        assertTrue(keyMapper.isSet(15, key));

        // b = U+0062 = 0000 0000 0110 0010
        assertFalse(keyMapper.isSet(16, key));
        assertFalse(keyMapper.isSet(17, key));
        assertFalse(keyMapper.isSet(18, key));
        assertFalse(keyMapper.isSet(19, key));

        assertFalse(keyMapper.isSet(20, key));
        assertFalse(keyMapper.isSet(21, key));
        assertFalse(keyMapper.isSet(22, key));
        assertFalse(keyMapper.isSet(23, key));

        assertFalse(keyMapper.isSet(24, key));
        assertTrue(keyMapper.isSet(25, key));
        assertTrue(keyMapper.isSet(26, key));
        assertFalse(keyMapper.isSet(27, key));

        assertFalse(keyMapper.isSet(28, key));
        assertFalse(keyMapper.isSet(29, key));
        assertTrue(keyMapper.isSet(30, key));
        assertFalse(keyMapper.isSet(31, key));

        // c = U+0063 = 0000 0000 0110 0011
        assertFalse(keyMapper.isSet(32, key));
        assertFalse(keyMapper.isSet(33, key));
        assertFalse(keyMapper.isSet(34, key));
        assertFalse(keyMapper.isSet(35, key));

        assertFalse(keyMapper.isSet(36, key));
        assertFalse(keyMapper.isSet(37, key));
        assertFalse(keyMapper.isSet(38, key));
        assertFalse(keyMapper.isSet(39, key));

        assertFalse(keyMapper.isSet(40, key));
        assertTrue(keyMapper.isSet(41, key));
        assertTrue(keyMapper.isSet(42, key));
        assertFalse(keyMapper.isSet(43, key));

        assertFalse(keyMapper.isSet(44, key));
        assertFalse(keyMapper.isSet(45, key));
        assertTrue(keyMapper.isSet(46, key));
        assertTrue(keyMapper.isSet(47, key));
    }

    @Test
    public void testNullKeyMap() {
        PatriciaTrie.KeyMapper<String> keyMapper = new PatriciaTrie.StringKeyMapper();
        assertFalse(keyMapper.isSet(0, null));
        assertFalse(keyMapper.isSet(100, null));
        assertFalse(keyMapper.isSet(1000, null));
    }

    @Test
    public void testEmptyKeyMap() {
        PatriciaTrie.KeyMapper<String> keyMapper = new PatriciaTrie.StringKeyMapper();
        // Note: this is a special case handled in PatriciaTrie
        assertTrue(keyMapper.isSet(0, ""));
        assertTrue(keyMapper.isSet(100, ""));
        assertTrue(keyMapper.isSet(1000, ""));
    }

    @Test
    public void testOverflowBit() {
        PatriciaTrie.KeyMapper<String> keyMapper = new PatriciaTrie.StringKeyMapper();
        String key = "a";

        // a = U+0061 = 0000 0000 0110 0001
        assertFalse(keyMapper.isSet(0, key));
        assertFalse(keyMapper.isSet(1, key));
        assertFalse(keyMapper.isSet(2, key));
        assertFalse(keyMapper.isSet(3, key));

        assertFalse(keyMapper.isSet(4, key));
        assertFalse(keyMapper.isSet(5, key));
        assertFalse(keyMapper.isSet(6, key));
        assertFalse(keyMapper.isSet(7, key));

        assertFalse(keyMapper.isSet(8, key));
        assertTrue(keyMapper.isSet(9, key));
        assertTrue(keyMapper.isSet(10, key));
        assertFalse(keyMapper.isSet(11, key));

        assertFalse(keyMapper.isSet(12, key));
        assertFalse(keyMapper.isSet(13, key));
        assertFalse(keyMapper.isSet(14, key));
        assertTrue(keyMapper.isSet(15, key));

        // Asking for overflow bits should return 1
        assertTrue(keyMapper.isSet(16, key));
        assertTrue(keyMapper.isSet(17, key));
        assertTrue(keyMapper.isSet(100, key));
    }
}
