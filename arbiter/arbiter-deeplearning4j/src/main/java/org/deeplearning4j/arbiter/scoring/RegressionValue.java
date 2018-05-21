package org.deeplearning4j.arbiter.scoring;

/**
 * Enumeration used to select the type of regression statistics to optimize on, with the various regression score functions
 * - MSE: mean squared error<br>
 * - MAE: mean absolute error<br>
 * - RMSE: root mean squared error<br>
 * - RSE: relative squared error<br>
 * - CorrCoeff: correlation coefficient<br>
 *
 * @deprecated Use {@link org.deeplearning4j.eval.RegressionEvaluation.Metric}
 */
@Deprecated
public enum RegressionValue {
    MSE, MAE, RMSE, RSE, CorrCoeff
}
