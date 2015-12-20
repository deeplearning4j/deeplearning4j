package org.arbiter.optimize.api.evaluation;

import org.arbiter.optimize.api.data.DataProvider;

import java.io.Serializable;

/**ModelEvaluator: Used to conduct additional evaluation.
 * For example, this may be classification performance on a test set or similar
 */
public interface ModelEvaluator<M,D,A> extends Serializable{
    A evaluateModel(M model, DataProvider<D> dataProvider);
}
