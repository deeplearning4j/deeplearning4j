/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform;

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
    Add, Subtract, Multiply, Divide, Modulus, ReverseSubtract, ReverseDivide, ScalarMin, ScalarMax
}
