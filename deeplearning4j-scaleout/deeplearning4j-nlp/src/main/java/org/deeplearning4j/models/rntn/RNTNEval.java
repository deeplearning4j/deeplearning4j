/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.models.rntn;

import org.deeplearning4j.eval.ConfusionMatrix;
import org.deeplearning4j.models.featuredetectors.autoencoder.recursive.Tree;
import org.nd4j.linalg.factory.Nd4j;
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
    private static final Logger log = LoggerFactory.getLogger(RNTNEval.class);


    /**
     * Eval the RNTN
     * @param rntn
     * @param trees
     */
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
        int predictionLabel = Nd4j.getBlasWrapper().iamax(tree.prediction());
        cf.add(treeGoldLabel,predictionLabel);
    }


    /**
     * Print the summary of the rntnresults
     * @return the summary of the rntn
     */
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
