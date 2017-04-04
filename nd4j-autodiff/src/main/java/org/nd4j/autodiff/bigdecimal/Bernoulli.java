package org.nd4j.autodiff.bigdecimal;

import java.util.*;
import java.math.*;


public class Bernoulli {
    /*
     * The list of all Bernoulli numbers as a vector, n=0,2,4,....
     */
    static Vector<Rational> a = new Vector<Rational>();

    public Bernoulli() {
        if (a.size() == 0) {
            a.add(Rational.ONE);
            a.add(new Rational(1, 6));
        }
    }


    protected void set(final int n, final Rational value) {
        final int nindx = n / 2;
        if (nindx < a.size())
            a.set(nindx, value);
        else {
            while (a.size() < nindx)
                a.add(Rational.ZERO);
            a.add(value);
        }
    }


    public Rational at(int n) {
        if (n == 1)
            return (new Rational(-1, 2));
        else if (n % 2 != 0)
            return Rational.ZERO;
        else {
            final int nindx = n / 2;
            if (a.size() <= nindx) {
                for (int i = 2 * a.size(); i <= n; i += 2)
                    set(i, doubleSum(i));
            }
            return a.elementAt(nindx);
        }
    }

    /*
     * Generate a new B_n by a standard double sum.
     *
     * @param n The index of the Bernoulli number.
     *
     * @return The Bernoulli number at n.
     */
    private Rational doubleSum(int n) {
        Rational resul = Rational.ZERO;
        for (int k = 0; k <= n; k++) {
            Rational jsum = Rational.ZERO;
            BigInteger bin = BigInteger.ONE;
            for (int j = 0; j <= k; j++) {
                BigInteger jpown = (new BigInteger("" + j)).pow(n);
                if (j % 2 == 0)
                    jsum = jsum.add(bin.multiply(jpown));
                else
                    jsum = jsum.subtract(bin.multiply(jpown));

                /*
                 * update binomial(k,j) recursively
                 */
                bin = bin.multiply(new BigInteger("" + (k - j)))
                        .divide(new BigInteger("" + (j + 1)));
            }
            resul = resul.add(jsum.divide(new BigInteger("" + (k + 1))));
        }
        return resul;
    }

} /* Bernoulli */
