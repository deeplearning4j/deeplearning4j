/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.text.corpora.treeparser;

import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.deeplearning4j.text.corpora.treeparser.transformer.TreeTransformer;

import java.util.ArrayList;
import java.util.List;

/**
 * Collapse unaries such that the
 * tree is only made of preterminals and leaves.
 *
 * @author Adam Gibson
 */
public class CollapseUnaries implements TreeTransformer {


    @Override
    public Tree transform(Tree tree) {
        if (tree.isPreTerminal() || tree.isLeaf()) {
            return tree;
        }

        List<Tree> children = tree.children();
        while (children.size() == 1 && !children.get(0).isLeaf()) {
            children = children.get(0).children();
        }

        List<Tree> processed = new ArrayList<>();
        for (Tree child : children)
            processed.add(transform(child));

        Tree ret = new Tree(tree);
        ret.connect(processed);

        return ret;
    }



}
