/*
 *  ******************************************************************************
 *  *
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

package com.atilika.kuromoji.trie;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import java.io.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class DoubleArrayTrieTest extends BaseDL4JTest {

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
