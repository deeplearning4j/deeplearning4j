package org.deeplearning4j.spark.impl.multilayer.evaluation;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.IEvaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * Reduction function for use with {@link IEvaluateFlatMapFunction} for distributed evaluation
 *
 * @author Alex Black
 */
@Slf4j
public class IEvaluationReduceFunction<T extends IEvaluation> implements Function2<T[], T[], T[]> {
    public IEvaluationReduceFunction() {}

    @Override
    public T[] call(T[] eval1, T[] eval2) throws Exception {
        for (int i = 0; i < eval1.length; i++) {
            eval1[i].merge(eval2[i]);
        }
        return eval1;
    }
}
