package org.deeplearning4j.spark.impl.multilayer.evaluation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.eval.IEvaluation;

/**
 * A simple function to merge IEvaluation instances
 *
 * @author Alex Black
 */
public class IEvaluateAggregateFunction<T extends IEvaluation> implements Function2<T[], T[], T[]> {
    @Override
    public T[] call(T[] v1, T[] v2) throws Exception {
        if (v1 == null)
            return v2;
        if (v2 == null)
            return v1;
        for (int i = 0; i < v1.length; i++) {
            v1[i].merge(v2[i]);
        }
        return v1;
    }
}
