package org.deeplearning4j.eval;

/**
 * The averaging approach for binary valuation measures when applied to multiclass classification problems.
 * Macro averaging: weight each class equally<br>
 * Micro averaging: weight each example equally<br>
 * Generally, macro averaging is preferred for imbalanced datasets
 *
 * @author Alex Black
 */
public enum EvaluationAveraging {
    Macro, Micro
}
