package org.deeplearning4j.text.treeparser;

import org.deeplearning4j.rntn.Tree;
import org.deeplearning4j.text.treeparser.transformer.TreeTransformer;
import org.deeplearning4j.word2vec.Word2Vec;

import java.util.ArrayList;
import java.util.List;

/**
 * Tree vectorization pipeline. Takes a word vector model (as a lookup table)
 * and a parser and handles vectorization of strings appropriate for an RNTN
 *
 * @author Adam Gibson
 */
public class TreeVectorizer {
    private Word2Vec vec;
    private TreeParser parser;
    private TreeTransformer treeTransformer = new BinarizeTreeTransformer();
    private TreeTransformer cnfTransformer = new CNFTransformer();

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
        for(Tree t : baseTrees) {
            ret.add(treeTransformer.transform(cnfTransformer.transform(t)));
        }

        return ret;

    }






}
