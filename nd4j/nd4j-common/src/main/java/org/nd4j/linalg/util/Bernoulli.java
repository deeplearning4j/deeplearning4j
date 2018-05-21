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

/*
 * To change this template, choose Tools | Templates and open the template in the editor.
 */

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

/**
 * Bernoulli numbers.
 */

class Bernoulli {
    /*
     * The list of all Bernoulli numbers as a vector, n=0,2,4,....
     */

    static List<Rational> a = new ArrayList<Rational>();

    public Bernoulli() {
        if (a.isEmpty()) {
            a.add(Rational.ONE);
            a.add(new Rational(1, 6));
        }
    }

    /**
     * Set a coefficient in the internal table.
     *
     * @param n     the zero-based index of the coefficient. n=0 for the constant term.
     * @param value the new value of the coefficient.
     */
    protected void set(final int n, final Rational value) {
        final int nindx = n / 2;
        if (nindx < a.size()) {
            a.set(nindx, value);
        } else {
            while (a.size() < nindx) {
                a.add(Rational.ZERO);
            }
            a.add(value);
        }
    }

    /**
     * The Bernoulli number at the index provided.
     *
     * @param n the index, non-negative.
     * @return the B_0=1 for n=0, B_1=-1/2 for n=1, B_2=1/6 for n=2 etc
     */
    public Rational at(int n) {
        if (n == 1) {
            return (new Rational(-1, 2));
        } else if (n % 2 != 0) {
            return Rational.ZERO;
        } else {
            final int nindx = n / 2;
            if (a.size() <= nindx) {
                for (int i = 2 * a.size(); i <= n; i += 2) {
                    set(i, doubleSum(i));
                }
            }
            return a.get(nindx);
        }
    }
    /* Generate a new B_n by a standard double sum.
     * @param n The index of the Bernoulli number.
     * @return The Bernoulli number at n.
     */

    private Rational doubleSum(int n) {
        Rational resul = Rational.ZERO;
        for (int k = 0; k <= n; k++) {
            Rational jsum = Rational.ZERO;
            BigInteger bin = BigInteger.ONE;
            for (int j = 0; j <= k; j++) {
                BigInteger jpown = BigInteger.valueOf(j).pow(n);
                if (j % 2 == 0) {
                    jsum = jsum.add(bin.multiply(jpown));
                } else {
                    jsum = jsum.subtract(bin.multiply(jpown));
                }
                /* update binomial(k,j) recursively
                 */
                bin = bin.multiply(BigInteger.valueOf(k - j)).divide(BigInteger.valueOf(j + 1));
            }
            resul = resul.add(jsum.divide(BigInteger.valueOf(k + 1)));
        }
        return resul;
    }
} /* Bernoulli */
