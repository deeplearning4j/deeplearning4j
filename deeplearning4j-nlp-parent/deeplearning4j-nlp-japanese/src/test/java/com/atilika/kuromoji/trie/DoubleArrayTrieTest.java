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

import java.io.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DoubleArrayTrieTest {

    @Test
    public void testSparseTrie() throws IOException {
        testSimpleTrie(false);
    }

    @Test
    public void testCompactTrie() throws IOException {
        testSimpleTrie(false);
    }

    private void testSimpleTrie(boolean compact) throws IOException {
        Trie trie = makeTrie();
        File costsFile = File.createTempFile("kuromoji-doublearraytrie-", ".bin");
        costsFile.deleteOnExit();

        DoubleArrayTrie doubleArrayTrie = new DoubleArrayTrie(compact);
        doubleArrayTrie.build(trie);

        OutputStream output = new FileOutputStream(costsFile);
        doubleArrayTrie.write(output);
        output.close();

        doubleArrayTrie = DoubleArrayTrie.read(new FileInputStream(costsFile));

        assertEquals(0, doubleArrayTrie.lookup("a"));
        assertTrue(doubleArrayTrie.lookup("abc") > 0);
        assertTrue(doubleArrayTrie.lookup("あいう") > 0);
        assertTrue(doubleArrayTrie.lookup("xyz") < 0);
    }

    private Trie makeTrie() {
        Trie trie = new Trie();
        trie.add("abc");
        trie.add("abd");
        trie.add("あああ");
        trie.add("あいう");
        return trie;
    }
}
