package org.deeplearning4j.spark.impl.multilayer.evaluation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.eval.ROCMultiClass;

/**
 * @author Alex Black
 */
public class ROCMultiClassReduceFunction implements Function2<ROCMultiClass, ROCMultiClass, ROCMultiClass> {
    @Override
    public ROCMultiClass call(ROCMultiClass roc1, ROCMultiClass roc2) throws Exception {
        roc1.merge(roc2);
        return roc1;
    }
}
