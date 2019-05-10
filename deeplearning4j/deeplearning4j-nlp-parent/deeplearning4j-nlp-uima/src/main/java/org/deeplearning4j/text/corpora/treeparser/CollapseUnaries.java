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
