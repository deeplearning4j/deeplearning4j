/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.sparse;

/**
 * Algebraic semiring to use for GraphBLAS-style sparse matrix operations
 * ({@code csr_spmv_semiring}, {@code csr_spmm_semiring}).
 *
 * <p>Each semiring defines three things: a multiplicative binary op, an additive binary op,
 * and an additive identity element.  For row i the operation is:
 * <pre>
 *   out[i] = REDUCE_{k in row i}( add(acc, mul(values[k], x[colIdx[k]])) )
 * </pre>
 * starting from the additive identity.
 *
 * <table>
 *   <tr><th>Enum</th><th>mul</th><th>add</th><th>identity</th></tr>
 *   <tr><td>PLUS_TIMES</td><td>×</td><td>+</td><td>0</td></tr>
 *   <tr><td>MIN_PLUS</td><td>+</td><td>min</td><td>+∞</td></tr>
 *   <tr><td>MAX_PLUS</td><td>+</td><td>max</td><td>-∞</td></tr>
 *   <tr><td>OR_AND</td><td>∧</td><td>∨</td><td>0 (false)</td></tr>
 *   <tr><td>MIN_TIMES</td><td>×</td><td>min</td><td>+∞</td></tr>
 * </table>
 */
public enum Semiring {
    /** Standard (Σ, ×): y[i] = sum_k values[k] * x[j] — identical to plain SpMV. */
    PLUS_TIMES(0),
    /** Tropical min-plus: y[i] = min_k (values[k] + x[j])  — used for SSSP/BFS distance. */
    MIN_PLUS(1),
    /** Max-plus: y[i] = max_k (values[k] + x[j]). */
    MAX_PLUS(2),
    /** Logical or-and: y[i] = OR_k (values[k] AND x[j])  — used for BFS frontier. */
    OR_AND(3),
    /** Min-times: y[i] = min_k (values[k] * x[j])  — probabilistic paths. */
    MIN_TIMES(4);

    private final int code;
    Semiring(int code) { this.code = code; }
    /** Returns the integer code passed as IArg to the native op. */
    public int code() { return code; }
    /** Look up a Semiring by its integer code. */
    public static Semiring fromCode(int code) {
        for (Semiring s : values()) if (s.code == code) return s;
        throw new IllegalArgumentException("Unknown semiring code: " + code);
    }
}
