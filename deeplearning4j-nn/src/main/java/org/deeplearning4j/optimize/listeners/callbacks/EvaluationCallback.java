package org.deeplearning4j.optimize.listeners.callbacks;

import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;

/**
 * This interface describes callback, which can be used with EvaluativeListener, to extend its functionality.
 *
 * PLEASE NOTE: This callback will be invoked AFTER evaluation finished for all evaluators.
 *
 * @author raver119@gmail.com
 */
public interface EvaluationCallback {

    void call(EvaluativeListener listener, Model model, long invocationsCount, IEvaluation[] evaluations);
}
