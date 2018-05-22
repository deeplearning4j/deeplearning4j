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

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.deeplearning4j.text.corpora.treeparser.transformer.TreeTransformer;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Binarizes trees.
 * Based on the work by Manning et. al in stanford corenlp
 *
 * @author Adam Gibson
 */
public class BinarizeTreeTransformer implements TreeTransformer {

    private String factor = "left";
    private int horizontonalMarkov = 999;



    private static final Logger log = LoggerFactory.getLogger(BinarizeTreeTransformer.class);

    @Override
    public Tree transform(Tree t) {
        if (t == null)
            return null;
        Deque<Pair<Tree, String>> stack = new ArrayDeque<>();
        stack.add(new Pair<>(t, t.label()));
        String originalLabel = t.label();
        while (!stack.isEmpty()) {
            Pair<Tree, String> curr = stack.pop();
            Tree node = curr.getFirst();

            for (Tree child : node.children())
                stack.add(new Pair<>(child, curr.getSecond()));


            if (node.children().size() > 2) {

                List<String> children = new ArrayList<>();
                for (int i = 0; i < node.children().size(); i++)
                    children.add(node.children().get(i).label());

                Tree copy = node.clone();
                //clear out children
                node.children().clear();

                Tree currNode = node;

                for (int i = 1; i < children.size() - 1; i++) {
                    if (factor.equals("right")) {
                        Tree newNode = new Tree(currNode);

                        List<String> subChildren =
                                        children.subList(i, Math.min(i + horizontonalMarkov, children.size()));

                        newNode.setLabel(originalLabel + "-" + "(" + StringUtils.join(subChildren, "-"));

                        newNode.setParent(currNode);

                        currNode.children().add(copy.children().remove(0));

                        currNode.firstChild().setParent(currNode);

                        currNode.children().add(newNode);

                        currNode = newNode;

                    } else {
                        Tree newNode = new Tree(currNode);

                        newNode.setParent(copy.firstChild());

                        List<String> childLabels =
                                        children.subList(Math.max(children.size() - i - horizontonalMarkov, 0), i);

                        Collections.reverse(childLabels);
                        newNode.setLabel(originalLabel + "-" + "(" + StringUtils.join(childLabels, "-"));

                        currNode.children().add(newNode);

                        currNode.firstChild().setParent(currNode);

                        currNode.children().add(copy.children().remove(copy.children().size() - 1));
                        currNode.lastChild().setParent(currNode);

                        currNode = newNode;
                    }
                }

                currNode.children().addAll(new ArrayList<>(copy.children()));
            }
        }

        addPreTerminal(t);
        return t;
    }

    private void addPreTerminal(Tree t) {
        if (t.isLeaf()) {
            Tree newLeaf = new Tree(t);
            newLeaf.setLabel(t.value());
            t.children().add(newLeaf);
            newLeaf.setParent(t);
        } else {
            for (Tree child : t.children())
                addPreTerminal(child);
        }
    }


    private void checkState(Tree tree, Set<Tree> nonBinarized) {
        for (Tree t : tree.children()) {
            checkState(t, nonBinarized);
        }

        if (tree.children().size() > 2) {
            Tree parent = tree.parent();
            if (parent == null)
                return;
            nonBinarized.add(tree);

        }
    }



}
