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

import org.junit.BeforeClass;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class NodeTest {

    @BeforeClass
    public static void setUpBeforeClass() throws Exception {}

    @Test
    public void testNode() {
        Trie trie = new Trie();

        Trie.Node node = trie.new Node('!');
        assertEquals('!', node.getKey());

        node = trie.new Node('1');
        assertEquals('1', node.getKey());

        node = trie.new Node('a');
        assertEquals('a', node.getKey());

        node = trie.new Node('！');
        assertEquals('！', node.getKey());

        node = trie.new Node('１');
        assertEquals('１', node.getKey());

        node = trie.new Node('あ');
        assertEquals('あ', node.getKey());

        node = trie.new Node('漢');
        assertEquals('漢', node.getKey());

    }

    @Test
    public void testAddChild() {
        Trie trie = new Trie();
        Trie.Node node = trie.new Node('a');

        Trie.Node returnedNode = node.addChild(trie.new Node('b'));
        assertEquals('b', returnedNode.getKey());
        assertEquals(1, node.getChildren().size());
        assertEquals('b', node.getChildren().get(0).getKey());

        returnedNode = node.addChild(trie.new Node('c'));
        assertEquals('c', returnedNode.getKey());
        assertEquals(2, node.getChildren().size());
        assertEquals('c', node.getChildren().get(1).getKey());
    }

    @Test
    public void testAdd() {
        Trie trie = new Trie();

        Trie.Node node = trie.new Node('a');
        node.add("");
        assertEquals(0, node.getChildren().size());

        node = trie.new Node('a');
        node.add("b");
        assertEquals(1, node.getChildren().size());
        assertEquals('b', node.getChildren().get(0).getKey());

        node = trie.new Node('a');
        node.add("bc");
        Trie.Node b = node.getChildren().get(0);
        assertEquals(1, node.getChildren().size());
        assertEquals('b', b.getKey());
        assertEquals(1, b.getChildren().size());
        Trie.Node c = b.getChildren().get(0);
        assertEquals('c', c.getKey());
        assertEquals(0, c.getChildren().size());

        node.add("bd");
        b = node.getChildren().get(0);
        assertEquals(1, node.getChildren().size());
        assertEquals('b', b.getKey());
        assertEquals(2, b.getChildren().size());
        c = b.getChildren().get(0);
        assertEquals('c', c.getKey());
        assertEquals(0, c.getChildren().size());
        Trie.Node d = b.getChildren().get(1);
        assertEquals('d', d.getKey());
        assertEquals(0, d.getChildren().size());
    }


    @Test
    public void testGetkey() {
        Trie trie = new Trie();

        Trie.Node node = trie.new Node('!');
        assertEquals('!', node.getKey());

        node = trie.new Node('1');
        assertEquals('1', node.getKey());

        node = trie.new Node('a');
        assertEquals('a', node.getKey());

        node = trie.new Node('！');
        assertEquals('！', node.getKey());

        node = trie.new Node('１');
        assertEquals('１', node.getKey());

        node = trie.new Node('あ');
        assertEquals('あ', node.getKey());

        node = trie.new Node('漢');
        assertEquals('漢', node.getKey());
    }

    @Test
    public void testHasSinglePath() {
        Trie trie = new Trie();

        Trie.Node node = trie.new Node('a');
        node.add("bcd");
        assertEquals(true, node.hasSinglePath());

        node.add("bce");
        assertEquals(false, node.hasSinglePath());
    }

    @Test
    public void testGetChildren() {
        Trie trie = new Trie();

        Trie.Node node = trie.new Node('a');
        node.add("bcd");
        node.add("bde");
        node.add("xyz");

        assertEquals(2, node.getChildren().size());
        assertEquals('b', node.getChildren().get(0).getKey());
        assertEquals('x', node.getChildren().get(1).getKey());
    }
}
