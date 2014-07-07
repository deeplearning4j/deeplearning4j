package org.deeplearning4j.text.treeparser.transformer;

import org.deeplearning4j.rntn.Tree;

/**
 * Tree transformer
 * @author Adam Gibson
 */
public interface TreeTransformer {

    /**
     * Applies a transform to a tree
     * @param t the tree to transform
     * @return the transformed tree
     */
    Tree transform(Tree t);


}
