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
 * Tree vectorization pipeline. Takes a word vector model (as a lookup table)
 * and a parser and handles vectorization of strings appropriate for an RNTN
 *
 * @author Adam Gibson
 */
public class TreeVectorizer {

    private TreeParser parser;
    private TreeTransformer treeTransformer = new BinarizeTreeTransformer();
    private TreeTransformer cnfTransformer = new CollapseUnaries();

    /**
     * Uses the given parser and model
     * for vectorization of strings
     * @param parser the parser to use for converting
     * strings to trees
     */
    public TreeVectorizer(TreeParser parser) {
        this.parser = parser;
    }

    /**
     * Uses word vectors from the passed in word2vec model
     * @throws Exception
     */
    public TreeVectorizer() throws Exception {
        this(new TreeParser());
    }

    /**
     * Vectorizes the passed in sentences
     * @param sentences the sentences to convert to trees
     * @return a list of trees pre converted with CNF and
     * binarized and word vectors at the leaves of the trees
     *
     * @throws Exception
     */
    public List<Tree> getTrees(String sentences) throws Exception {
        List<Tree> ret = new ArrayList<>();
        List<Tree> baseTrees = parser.getTrees(sentences);
        for (Tree t : baseTrees) {
            Tree binarized = treeTransformer.transform(t);
            binarized = cnfTransformer.transform(binarized);
            ret.add(binarized);
        }

        return ret;

    }


    /**
     * Vectorizes the passed in sentences
     * @param sentences the sentences to convert to trees
     * @param label the label for the sentence
     * @param labels all of the possible labels for the trees
     * @return a list of trees pre converted with CNF and
     * binarized and word vectors at the leaves of the trees
     *
     * @throws Exception
     */
    public List<Tree> getTreesWithLabels(String sentences, String label, List<String> labels) throws Exception {
        List<Tree> ret = new ArrayList<>();
        List<Tree> baseTrees = parser.getTreesWithLabels(sentences, label, labels);
        for (Tree t : baseTrees) {
            Tree binarized = treeTransformer.transform(t);
            binarized = cnfTransformer.transform(binarized);
            ret.add(binarized);
        }

        return ret;

    }



    /**
     * Vectorizes the passed in sentences
     * @param sentences the sentences to convert to trees
     * @param labels all of the possible labels for the trees
     * @return a list of trees pre converted with CNF and
     * binarized and word vectors at the leaves of the trees
     *
     * @throws Exception
     */
    public List<Tree> getTreesWithLabels(String sentences, List<String> labels) throws Exception {
        List<String> realLabels = new ArrayList<>(labels);
        if (!realLabels.contains("NONE"))
            realLabels.add("NONE");
        List<Tree> ret = new ArrayList<>();
        List<Tree> baseTrees = parser.getTreesWithLabels(sentences, realLabels);
        for (Tree t : baseTrees) {
            Tree binarized = treeTransformer.transform(t);
            binarized = cnfTransformer.transform(binarized);
            ret.add(binarized);
        }

        return ret;

    }



}
