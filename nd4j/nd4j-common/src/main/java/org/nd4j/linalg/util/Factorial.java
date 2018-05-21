/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 *
 */

package org.nd4j.linalg.util;

/**
 *
 * @author dmtrl
 */

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * Factorials.
 */
class Factorial {

    /**
     * The list of all factorials as a vector.
     */
    static List<BigInteger> a = new ArrayList<>();

    /**
     * ctor().
     * Initialize the vector of the factorials with 0!=1 and 1!=1.
     */
    public Factorial() {
        if (a.isEmpty()) {
            a.add(BigInteger.ONE);
            a.add(BigInteger.ONE);
        }
    }

    /**
     * Compute the factorial of the non-negative integer.
     *
     * @param n the argument to the factorial, non-negative.
     * @return the factorial of n.
     */
    public BigInteger at(int n) {
        while (a.size() <= n) {
            final int lastn = a.size() - 1;
            final BigInteger nextn = BigInteger.valueOf(lastn + 1);
            a.add(a.get(lastn).multiply(nextn));
        }
        return a.get(n);
    }
} /* Factorial */
