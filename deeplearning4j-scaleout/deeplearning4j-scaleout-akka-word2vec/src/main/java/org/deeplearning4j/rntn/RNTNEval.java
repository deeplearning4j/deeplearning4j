package org.deeplearning4j.rntn;

import org.deeplearning4j.eval.ConfusionMatrix;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Set;

/**
 * Recursive counter for an RNTN
 *
 * @author Adam Gibson
 */
public class RNTNEval {

    private ConfusionMatrix<Integer> cf = new ConfusionMatrix<>();
    private static Logger log = LoggerFactory.getLogger(RNTNEval.class);



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
        int treeGoldLabel = tree.goldLabel();
        int predictionLabel = SimpleBlas.iamax(tree.prediction());
        cf.add(treeGoldLabel,predictionLabel);
    }


    public String stats() {
        StringBuilder builder = new StringBuilder()
                .append("\n");
        Set<Integer> classes = cf.getClasses();
        for(Integer clazz : classes) {
            for(Integer clazz2 : classes) {
                int count = cf.getCount(clazz, clazz2);
                if(count != 0)
                    builder.append("\nActual Class " + clazz + " was predicted with Predicted " + clazz2 + " with count " + count  + " times\n");
            }
        }
        return builder.toString();
    }

}
