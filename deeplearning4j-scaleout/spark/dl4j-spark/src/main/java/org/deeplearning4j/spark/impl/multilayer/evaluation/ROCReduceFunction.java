package org.deeplearning4j.spark.impl.multilayer.evaluation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.eval.ROC;

/**
 * @author Alex Black
 */
public class ROCReduceFunction implements Function2<ROC, ROC, ROC> {
    @Override
    public ROC call(ROC roc1, ROC roc2) throws Exception {
        roc1.merge(roc2);
        return roc1;
    }
}
