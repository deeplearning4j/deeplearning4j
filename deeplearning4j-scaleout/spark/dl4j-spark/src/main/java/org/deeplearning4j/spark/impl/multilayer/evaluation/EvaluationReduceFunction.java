package org.deeplearning4j.spark.impl.multilayer.evaluation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.eval.Evaluation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by nyghtowl on 1/13/16.
 */
public class EvaluationReduceFunction implements Function2<Evaluation, Evaluation, Evaluation> {

    protected static Logger log = LoggerFactory.getLogger(EvaluationReduceFunction.class);

    public EvaluationReduceFunction(){}

    @Override
    public Evaluation call(Evaluation eval1, Evaluation eval2) throws Exception {
        eval1.merge(eval2);
        return eval1;
    }
}
