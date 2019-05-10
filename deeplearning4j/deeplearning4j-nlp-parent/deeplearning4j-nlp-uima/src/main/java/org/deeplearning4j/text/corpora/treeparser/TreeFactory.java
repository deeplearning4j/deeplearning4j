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


import org.apache.uima.fit.util.FSCollectionFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.tcas.Annotation;
import org.cleartk.syntax.constituent.type.TreebankNode;
import org.cleartk.syntax.constituent.type.TreebankNodeUtil;
import org.cleartk.token.type.Token;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive.Tree;
import org.nd4j.linalg.collection.MultiDimensionalMap;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * Static movingwindow class handling the conversion of
 * treebank nodes to Trees useful
 * for recursive neural tensor networks
 *
 * @author Adam Gibson
 */
public class TreeFactory {


    private TreeFactory() {}

    /**
     * Builds a tree recursively
     * adding the children as necessary
     * @param node the node to build the tree based on
     * @param labels the labels to assign for each span
     * @return the compiled tree with all of its children
     * and childrens' children recursively
     * @throws Exception
     */
    public static Tree buildTree(TreebankNode node, Pair<String, MultiDimensionalMap<Integer, Integer, String>> labels,
                    List<String> possibleLabels) throws Exception {
        if (node.getLeaf())
            return toTree(node);
        else {
            List<TreebankNode> preChildren = children(node);
            List<Tree> children = new ArrayList<>();
            Tree t = toTree(node);
            for (Pair<Integer, Integer> interval : labels.getSecond().keySet()) {
                if (inRange(interval.getFirst(), interval.getSecond(), t)) {
                    t.setGoldLabel(possibleLabels
                                    .indexOf(labels.getSecond().get(interval.getFirst(), interval.getSecond())));
                    break;
                }
            }

            for (int i = 0; i < preChildren.size(); i++) {
                children.add(buildTree(preChildren.get(i)));
            }

            t.connect(children);
            return t;

        }
    }

    /**
     * Converts a treebank node to a tree
     * @param node the node to convert
     * @param labels the labels to assign for each span
     * @return the tree with the same tokens and type as
     * the given tree bank node
     * @throws Exception
     */
    public static Tree toTree(TreebankNode node, Pair<String, MultiDimensionalMap<Integer, Integer, String>> labels)
                    throws Exception {
        List<String> tokens = tokens(node);
        Tree ret = new Tree(tokens);
        ret.setValue(node.getNodeValue());
        ret.setLabel(node.getNodeType());
        ret.setType(node.getNodeType());
        ret.setBegin(node.getBegin());
        ret.setEnd(node.getEnd());
        ret.setParse(TreebankNodeUtil.toTreebankString(node));
        if (node.getNodeTags() != null)
            ret.setTags(tags(node));
        else
            ret.setTags(Arrays.asList(node.getNodeType()));
        return ret;
    }



    /**
     * Builds a tree recursively
     * adding the children as necessary
     * @param node the node to build the tree based on
     * @return the compiled tree with all of its children
     * and childrens' children recursively
     * @throws Exception
     */
    public static Tree buildTree(TreebankNode node) throws Exception {
        if (node.getLeaf())
            return toTree(node);
        else {
            List<TreebankNode> preChildren = children(node);
            List<Tree> children = new ArrayList<>();
            Tree t = toTree(node);
            for (int i = 0; i < preChildren.size(); i++) {
                children.add(buildTree(preChildren.get(i)));
            }

            t.connect(children);
            return t;

        }



    }

    /**
     * Converts a treebank node to a tree
     * @param node the node to convert
     * @return the tree with the same tokens and type as
     * the given tree bank node
     * @throws Exception
     */
    public static Tree toTree(TreebankNode node) throws Exception {
        List<String> tokens = tokens(node);
        Tree ret = new Tree(tokens);
        ret.setValue(node.getNodeValue());
        ret.setLabel(node.getNodeType());
        ret.setType(node.getNodeType());
        ret.setBegin(node.getBegin());
        ret.setEnd(node.getEnd());
        ret.setParse(TreebankNodeUtil.toTreebankString(node));
        if (node.getNodeTags() != null)
            ret.setTags(tags(node));
        else
            ret.setTags(Arrays.asList(node.getNodeType()));
        return ret;
    }


    private static List<String> tags(TreebankNode node) {
        List<String> ret = new ArrayList<>();
        for (int i = 0; i < node.getNodeTags().size(); i++)
            ret.add(node.getNodeTags(i));
        return ret;
    }


    private static List<TreebankNode> children(TreebankNode node) {
        return new ArrayList<>(FSCollectionFactory.create(node.getChildren(), TreebankNode.class));
    }

    private static List<String> tokens(Annotation ann) throws Exception {
        List<String> ret = new ArrayList<>();
        for (Token t : JCasUtil.select(ann.getCAS().getJCas(), Token.class)) {
            ret.add(t.getCoveredText());
        }
        return ret;
    }

    private static boolean inRange(int begin, int end, Tree tree) {
        return tree.getBegin() >= begin && tree.getEnd() <= end;
    }

}
