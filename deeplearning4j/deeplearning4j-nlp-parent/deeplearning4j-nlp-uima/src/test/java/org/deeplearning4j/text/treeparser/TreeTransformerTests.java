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

package org.deeplearning4j.text.treeparser;


import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.deeplearning4j.text.corpora.treeparser.BinarizeTreeTransformer;
import org.deeplearning4j.text.corpora.treeparser.CollapseUnaries;
import org.deeplearning4j.text.corpora.treeparser.TreeParser;
import org.deeplearning4j.text.corpora.treeparser.transformer.TreeTransformer;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 7/1/14.
 */
public class TreeTransformerTests {

    private static final Logger log = LoggerFactory.getLogger(TreeTransformerTests.class);
    private TreeParser parser;

    @Before
    public void init() throws Exception {
        parser = new TreeParser();
    }



    @Test
    public void testBinarize() throws Exception {
        List<Tree> trees = parser.getTrees("Is so sad for my apl friend. i missed the new moon trailer.");
        TreeTransformer t = new BinarizeTreeTransformer();
        TreeTransformer cnf = new CollapseUnaries();
        for (Tree t2 : trees) {
            t2 = t.transform(t2);
            assertChildSize(t2);
            for (Tree child : t2.children())
                if (child.isLeaf())
                    assertEquals("Found leaf node with parent that was not a preterminal", true, t2.isPreTerminal());
            t2 = cnf.transform(t2);
            assertCollapsedUnaries(t2);
        }
    }


    private void assertCollapsedUnaries(Tree tree) {
        for (Tree child : tree.children())
            assertCollapsedUnaries(child);
        if (tree.children().size() == 1 && !tree.isPreTerminal())
            throw new IllegalStateException("Trees with size of 1 and non preterminals should have been collapsed");
    }

    private void assertChildSize(Tree tree) {
        for (Tree child : tree.children()) {
            assertChildSize(child);
        }

        assertEquals("Tree is not valid " + tree + " tree children size was " + tree.children().size(), true,
                        tree.isLeaf() || tree.isPreTerminal() || tree.children().size() <= 2);


    }


}
