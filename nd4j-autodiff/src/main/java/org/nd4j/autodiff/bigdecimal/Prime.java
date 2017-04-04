package org.nd4j.autodiff.bigdecimal;

import java.math.BigInteger;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class Prime {

    private static final Logger LOGGER = LoggerFactory.getLogger(Prime.class);

    static ArrayList<BigInteger> a = new ArrayList<>();


    protected static BigInteger nMax = new BigInteger("-1");


    public Prime() {
        if (a.isEmpty()) {
            a.add(new BigInteger(Integer.toString(2)));
            a.add(new BigInteger(Integer.toString(3)));
            a.add(new BigInteger(Integer.toString(5)));
            a.add(new BigInteger(Integer.toString(7)));
            a.add(new BigInteger(Integer.toString(11)));
            a.add(new BigInteger(Integer.toString(13)));
            a.add(new BigInteger(Integer.toString(17)));
        }
        nMax = a.get(a.size() - 1);
    }


    public boolean contains(BigInteger n) {
        switch (millerRabin(n)) {
        case -1:
            return false;
        case 1:
            return true;
        }
        growto(n);
        return a.contains(n);
    }


    public boolean isSPP(final BigInteger n, final BigInteger a) {
        final BigInteger two = new BigInteger(Integer.toString(2));

        /*
         * numbers less than 2 are not prime
         */
        if (n.compareTo(two) == -1)
            return false;
        /*
         * 2 is prime
         */
        else if (n.compareTo(two) == 0)
            return true;
        /*
         * even numbers >2 are not prime
         */
        else if (n.remainder(two).compareTo(BigInteger.ZERO) == 0)
            return false;
        else {
            /*
             * q= n- 1 = d *2^s with d odd
             */
            final BigInteger q = n.subtract(BigInteger.ONE);
            int s = q.getLowestSetBit();
            BigInteger d = q.shiftRight(s);

            /*
             * test whether a^d = 1 (mod n)
             */
            if (a.modPow(d, n).compareTo(BigInteger.ONE) == 0)
                return true;

            /*
             * test whether a^(d*2^r) = -1 (mod n), 0<=r<s
             */
            for (int r = 0; r < s; r++) {
                if (a.modPow(d.shiftLeft(r), n).compareTo(q) == 0)
                    return true;
            }
            return false;
        }
    }


    public int millerRabin(final BigInteger n) {
        /*
         * list of limiting numbers which fail tests on k primes, A014233 in the
         * OEIS
         */
        final String[] mr = { "2047", "1373653", "25326001", "3215031751", "2152302898747",
                "3474749660383", "341550071728321" };
        int mrLim = 0;
        while (mrLim < mr.length) {
            int l = n.compareTo(new BigInteger(mr[mrLim]));
            if (l < 0)
                break;
            /*
             * if one of the pseudo-primes: this is a composite
             */
            else if (l == 0)
                return -1;
            mrLim++;
        }
        /*
         * cannot test candidates larger than the last in the mr list
         */
        if (mrLim == mr.length)
            return 0;

        /*
         * test the bases prime(1), prime(2) up to prime(mrLim+1)
         */
        for (int p = 0; p <= mrLim; p++)
            if (!isSPP(n, at(p)))
                return -1;
        return 1;
    }


    public BigInteger at(int i) {
        /*
         * If the current list is too small, increase in intervals of 5 until
         * the list has at least i elements.
         */
        while (i >= a.size()) {
            growto(nMax.add(new BigInteger(Integer.toString(5))));
        }
        return a.get(i);
    }


    public BigInteger pi(BigInteger n) {
        /*
         * If the current list is too small, increase in intervals of 5 until
         * the list has at least i elements.
         */
        growto(n);
        BigInteger r = new BigInteger("0");
        for (int i = 0; i < a.size(); i++)
            if (a.get(i).compareTo(n) <= 0)
                r = r.add(BigInteger.ONE);
        return r;
    }


    public BigInteger nextprime(BigInteger n) {
        /* if n <=1, return 2 */
        if (n.compareTo(BigInteger.ONE) <= 0)
            return a.get(0);

        /*
         * If the currently largest element in the list is too small, increase
         * in intervals of 5 until the list has at least i elements.
         */
        while (a.get(a.size() - 1).compareTo(n) <= 0) {
            growto(nMax.add(new BigInteger(Integer.toString(5))));
        }
        for (int i = 0; i < a.size(); i++)
            if (a.get(i).compareTo(n) == 1)
                return a.get(i);
        return a.get(a.size() - 1);
    }


    public BigInteger prevprime(BigInteger n) {
        /* if n <=2, return 0 */
        if (n.compareTo(BigInteger.ONE) <= 0)
            return BigInteger.ZERO;

        /*
         * If the currently largest element in the list is too small, increase
         * in intervals of 5 until the list has at least i elements.
         */
        while (a.get(a.size() - 1).compareTo(n) < 0)
            growto(nMax.add(new BigInteger(Integer.toString(5))));

        for (int i = 0; i < a.size(); i++)
            if (a.get(i).compareTo(n) >= 0)
                return a.get(i - 1);
        return a.get(a.size() - 1);
    }


    protected void growto(BigInteger n) {
        while (nMax.compareTo(n) < 0) {
            nMax = nMax.add(BigInteger.ONE);
            boolean isp = true;
            for (int p = 0; p < a.size(); p++) {
                /*
                 * Test the list of known primes only up to sqrt(n)
                 */
                if (a.get(p).multiply(a.get(p)).compareTo(nMax) == 1)
                    break;

                /*
                 * The next case means that the p'th number in the list of known
                 * primes divides nMax and nMax cannot be a prime.
                 */
                if (nMax.remainder(a.get(p)).compareTo(BigInteger.ZERO) == 0) {
                    isp = false;
                    break;
                }
            }
            if (isp)
                a.add(nMax);
        }
    }


    public static void main(String[] args) throws Exception {
        Prime a = new Prime();
        int n = (new Integer(args[0])).intValue();
        if (n >= 1) {
            if (n >= 2)
                LOGGER.debug("prime({}) = ", n - 1, a.at(n - 1));
            LOGGER.debug("prime(" + n + ") = " + a.at(n));
            LOGGER.debug("prime(" + (n + 1) + ") = " + a.at(n + 1));
            LOGGER.debug("pi({}) = {}", n, a.pi(new BigInteger(Integer.toString(n))));
        }
    }
} /* Prime */
