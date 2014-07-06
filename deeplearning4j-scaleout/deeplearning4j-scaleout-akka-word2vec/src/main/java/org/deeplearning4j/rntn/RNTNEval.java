package org.deeplearning4j.rntn;

import org.deeplearning4j.eval.ConfusionMatrix;
import org.jblas.SimpleBlas;

import java.util.List;

/**
 * Recursive counter for an RNTN
 *
 * @author Adam Gibson
 */
public class RNTNEval {

   private ConfusionMatrix<Integer> cf = new ConfusionMatrix<>();

    public void eval(RNTN rntn, List<Tree> trees) {
        for(Tree t : trees) {
            rntn.forwardPropagateTree(t);
            count(t);
        }

    }

    private void count(Tree tree) {
        if(tree.isLeaf())
            return;
        if(tree.prediction() == null) {
            return;
        }

        for(Tree t : tree.children())
            count(t);
        cf.add(SimpleBlas.iamax(tree.prediction()),tree.goldLabel());
    }


    public String stats() {
        return cf.toString();
    }


}
