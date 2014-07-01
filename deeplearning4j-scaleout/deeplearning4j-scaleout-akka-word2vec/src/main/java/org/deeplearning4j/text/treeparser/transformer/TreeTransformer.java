package org.deeplearning4j.text.treeparser.transformer;

import org.deeplearning4j.rntn.Tree;

/**
 * Tree transformer
 * @author Adam Gibson
 */
public interface TreeTransformer {

    Tree transform(Tree t);


}
