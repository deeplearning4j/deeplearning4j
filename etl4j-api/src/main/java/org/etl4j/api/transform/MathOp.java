package org.etl4j.api.transform;

/**
 * Mathematical operations for Double, Integer and Long columns<br>
 *
 * Add<br>
 * Subtract<br>
 * Multiply<br>
 * Divide<br>
 * Modulus<br>
 * Reverse subtract: do scalar - x (instead of x-scalar in Subtract)<br>
 * Reverse divide: do scalar/x (instead of x/scalar in Divide)<br>
 * Scalar min: return Min(scalar,x)<br>
 * Scalar max: return Max(scalar,x)<br>
 *
 * @author Alex Black
 */
public enum MathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulus,
    ReverseSubtract,
    ReverseDivide,
    ScalarMin,
    ScalarMax
}
