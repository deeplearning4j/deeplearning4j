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

import java.util.ArrayList;
import java.util.List;

/**
 * Simple Trie used to build the DoubleArrayTrie
 */
public class Trie {

    /** Root node */
    private Node root;

    /**
     * Constructor
     * <p>
     * Initialize this as an empty trie
     */
    public Trie() {
        root = new Node();
    }

    /**
     * Adds an input value to this trie
     * <p>
     * Before the value is added, a terminating character (U+0001) is appended to the input string
     *
     * @param value  value to add to this trie
     */
    public void add(String value) {
        root.add(value, true);
    }

    /**
     * Returns this trie's root node
     *
     * @return root node, not null
     */
    public Node getRoot() {
        return root;
    }

    /**
     * Trie Node
     */
    public class Node {
        private char key;

        private List<Node> children = new ArrayList<>();

        /**
         * Constructor
         */
        public Node() {}

        /**
         * Constructor
         *
         * @param key  this node's key
         */
        public Node(char key) {
            this.key = key;
        }

        /**
         * Add string to add to this node
         *
         * @param value  string value, not null
         */
        public void add(String value) {
            add(value, false);
        }

        public void add(String value, boolean terminate) {
            if (value.length() == 0) {
                return;
            }

            Node node = addChild(new Node(value.charAt(0)));

            for (int i = 1; i < value.length(); i++) {
                node = node.addChild(new Node(value.charAt(i)));
            }

            if (terminate && (node != null)) {
                node.addChild(new Node(DoubleArrayTrie.TERMINATING_CHARACTER));
            }
        }

        /**
         * Adds a new child node to this node
         *
         * @param newNode  new child to add
         * @return the child node added, or, if a node with same key already exists, that node
         */
        public Node addChild(Node newNode) {
            Node child = getChild(newNode.getKey());
            if (child == null) {
                children.add(newNode);
                child = newNode;
            }
            return child;
        }

        /**
         * Return this node's key
         *
         * @return key
         */
        public char getKey() {
            return key;
        }

        /**
         * Predicate indicating if children following this node forms single key path (no branching)
         * <p>
         * For example, if we have "abcde" and "abfgh" in the trie, calling this method on node "a" and "b" returns false.
         * However, this method on "c", "d", "e", "f", "g" and "h" returns true.
         *
         * @return true if this node has a single key path. false otherwise.
         */
        public boolean hasSinglePath() {
            switch (children.size()) {
                case 0:
                    return true;
                case 1:
                    return children.get(0).hasSinglePath();
                default:
                    return false;
            }
        }

        /**
         * Returns this node's child nodes
         *
         * @return child nodes, not null
         */
        public List<Node> getChildren() {
            return children;
        }

        /**
         * Searches this nodes for a child with a specific key
         *
         * @param key  key to search for
         * @return node matching the input key if it exists, otherwise null
         */
        private Node getChild(char key) {
            for (Node child : children) {
                if (child.getKey() == key) {
                    return child;
                }
            }
            return null;
        }
    }
}
