package org.deeplearning4j.text.corpora.treeparser.transformer;

import org.deeplearning4j.models.rntn.Tree;

/**
 * Tree transformer
 * @author Adam Gibson
 */
public interface TreeTransformer {

    /**
     * Applies a applyTransformToOrigin to a tree
     * @param t the tree to applyTransformToOrigin
     * @return the transformed tree
     */
    Tree transform(Tree t);


}
