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

import com.atilika.kuromoji.trie.Trie.Node;
import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import static org.junit.Assert.*;

public class TrieTest extends BaseDL4JTest {

    @Test
    public void testGetRoot() {
        Trie trie = new Trie();
        Node rootNode = trie.getRoot();
        assertNotNull(rootNode);
    }

    @Test
    public void testAdd() {
        Trie trie = new Trie();
        trie.add("aa");
        trie.add("ab");
        trie.add("bb");

        Node rootNode = trie.getRoot();
        assertEquals(2, rootNode.getChildren().size());
        assertEquals(2, rootNode.getChildren().get(0).getChildren().size());
        assertEquals(1, rootNode.getChildren().get(1).getChildren().size());
    }

    @Test
    public void testGetChildren() {
        Trie trie = new Trie();
        trie.add("aa");
        trie.add("ab");
        trie.add("bb");

        Node rootNode = trie.getRoot();
        assertEquals(2, rootNode.getChildren().size());
        assertEquals(2, rootNode.getChildren().get(0).getChildren().size());
        assertEquals(1, rootNode.getChildren().get(1).getChildren().size());
    }

    @Test
    public void testSinglePath() {
        Trie trie = new Trie();
        assertTrue(trie.getRoot().hasSinglePath());
        trie.add("abcdef");
        assertTrue(trie.getRoot().hasSinglePath());
        trie.add("abdfg");
        Node rootNode = trie.getRoot();
        assertEquals(2, rootNode.getChildren().get(0).getChildren().get(0).getChildren().size());
        assertTrue(rootNode.getChildren().get(0).getChildren().get(0).getChildren().get(0).hasSinglePath());
        assertTrue(rootNode.getChildren().get(0).getChildren().get(0).getChildren().get(1).hasSinglePath());
    }
}
