package org.nd4j.linalg.util;

/**
 *
 * @author dmtrl
 */
import java.math.*;
import java.util.Vector;

/** Factorials.
 */
class Factorial {

    /** The list of all factorials as a vector.
     */
    static Vector<BigInteger> a = new Vector<BigInteger>();

    /** ctor().
     * Initialize the vector of the factorials with 0!=1 and 1!=1.
     */
    public Factorial() {
        if (a.size() == 0) {
            a.add(BigInteger.ONE);
            a.add(BigInteger.ONE);
        }
    }

    /** Compute the factorial of the non-negative integer.
     * @param n the argument to the factorial, non-negative.
     * @return the factorial of n.
     */
    public BigInteger at(int n) {
        while (a.size() <= n) {
            final int lastn = a.size() - 1;
            final BigInteger nextn = new BigInteger("" + (lastn + 1));
            a.add(a.elementAt(lastn).multiply(nextn));
        }
        return a.elementAt(n);
    }
} /* Factorial */