package org.nd4j.autodiff.bigdecimal;

import java.math.BigInteger;
import java.util.Collections;
import java.util.Vector;


public class Ifactor implements Cloneable, Comparable<Ifactor> {

    public BigInteger n;

    /*
     * The bases and powers of the prime factorization. representation n =
     * primeexp[0]^primeexp[1]*primeexp[2]^primeexp[3]*... The value 0 is
     * represented by an empty vector, the value 1 by a vector of length 1 with
     * a single power of 0.
     */
    public Vector<Integer> primeexp;

    final public static Ifactor ONE = new Ifactor(1);

    final public static Ifactor ZERO = new Ifactor(0);


    public Ifactor(int number) {
        n = new BigInteger("" + number);
        primeexp = new Vector<Integer>();
        if (number > 1) {
            int primindx = 0;
            Prime primes = new Prime();
            /*
             * Test division against all primes.
             */
            while (number > 1) {
                int ex = 0;
                /*
                 * primindx=0 refers to 2, =1 to 3, =2 to 5, =3 to 7 etc
                 */
                int p = primes.at(primindx).intValue();
                while (number % p == 0) {
                    ex++;
                    number /= p;
                    if (number == 1)
                        break;
                }
                if (ex > 0) {
                    primeexp.add(new Integer(p));
                    primeexp.add(new Integer(ex));
                }
                primindx++;
            }
        } else if (number == 1) {
            primeexp.add(new Integer(1));
            primeexp.add(new Integer(0));
        }
    } /* Ifactor */


    public Ifactor(BigInteger number) {
        n = number;
        primeexp = new Vector<Integer>();
        if (number.compareTo(BigInteger.ONE) == 0) {
            primeexp.add(new Integer(1));
            primeexp.add(new Integer(0));
        } else {
            int primindx = 0;
            Prime primes = new Prime();
            /*
             * Test for division against all primes.
             */
            while (number.compareTo(BigInteger.ONE) == 1) {
                int ex = 0;
                BigInteger p = primes.at(primindx);
                while (number.remainder(p).compareTo(BigInteger.ZERO) == 0) {
                    ex++;
                    number = number.divide(p);
                    if (number.compareTo(BigInteger.ONE) == 0)
                        break;
                }
                if (ex > 0) {
                    primeexp.add(new Integer(p.intValue()));
                    primeexp.add(new Integer(ex));
                }
                primindx++;
            }
        }
    } /* Ifactor */


    public Ifactor(Vector<Integer> pows) {
        primeexp = new Vector<Integer>(2 * pows.size());
        if (pows.size() > 0) {
            n = BigInteger.ONE;
            Prime primes = new Prime();
            /*
             * Build the full number by the product of all powers of the primes.
             */
            for (int primindx = 0; primindx < pows.size(); primindx++) {
                int ex = pows.elementAt(primindx).intValue();
                final BigInteger p = primes.at(primindx);
                n = n.multiply(p.pow(ex));
                primeexp.add(new Integer(p.intValue()));
                primeexp.add(new Integer(ex));
            }
        } else
            n = BigInteger.ZERO;
    } /* Ifactor */


    public Ifactor(Ifactor oth) {
        n = oth.n;
        primeexp = oth.primeexp;
    } /* Ifactor */


    public Ifactor clone() {
        Vector<Integer> p = (Vector<Integer>) primeexp.clone();
        Ifactor cl = new Ifactor(0);
        cl.n = new BigInteger("" + n);
        return cl;
    } /* Ifactor.clone */


    public boolean equals(final Ifactor oth) {
        return (n.compareTo(oth.n) == 0);
    } /* Ifactor.equals */


    public Ifactor multiply(final BigInteger oth) {
        /*
         * the optimization is to factorize oth _before_ multiplying
         */
        return (multiply(new Ifactor(oth)));
    } /* Ifactor.multiply */


    public Ifactor multiply(final int oth) {
        /*
         * the optimization is to factorize oth _before_ multiplying
         */
        return (multiply(new Ifactor(oth)));
    } /* Ifactor.multiply */


    public Ifactor multiply(final Ifactor oth) {
        /*
         * This might be done similar to the lcm() implementation by adding the
         * powers of the components and calling the constructor with the list of
         * exponents. This here is the simplest implementation, but slow because
         * it calls another prime factorization of the product: return( new
         * Ifactor(n.multiply(oth.n))) ;
         */
        return multGcdLcm(oth, 0);
    }


    public Ifactor lcm(final Ifactor oth) {
        return multGcdLcm(oth, 2);
    }


    public Ifactor gcd(final Ifactor oth) {
        return multGcdLcm(oth, 1);
    }


    protected Ifactor multGcdLcm(final Ifactor oth, int type) {
        Ifactor prod = new Ifactor(0);
        /*
         * skip the case where 0*something =0, falling thru to the empty
         * representation for 0
         */
        if (primeexp.size() != 0 && oth.primeexp.size() != 0) {
            /*
             * Cases of 1 times something return something. Cases of lcm(1,
             * something) return something. Cases of gcd(1, something) return 1.
             */
            if (primeexp.firstElement().intValue() == 1 && type == 0)
                return oth;
            else if (primeexp.firstElement().intValue() == 1 && type == 2)
                return oth;
            else if (primeexp.firstElement().intValue() == 1 && type == 1)
                return this;
            else if (oth.primeexp.firstElement().intValue() == 1 && type == 0)
                return this;
            else if (oth.primeexp.firstElement().intValue() == 1 && type == 2)
                return this;
            else if (oth.primeexp.firstElement().intValue() == 1 && type == 1)
                return oth;
            else {
                int idxThis = 0;
                int idxOth = 0;
                switch (type) {
                case 0:
                    prod.n = n.multiply(oth.n);
                    break;
                case 1:
                    prod.n = n.gcd(oth.n);
                    break;
                case 2:
                    /*
                     * the awkward way, lcm = product divided by gcd
                     */
                    prod.n = n.multiply(oth.n).divide(n.gcd(oth.n));
                    break;
                }

                /*
                 * scan both representations left to right, increasing prime
                 * powers
                 */
                while (idxOth < oth.primeexp.size() || idxThis < primeexp.size()) {
                    if (idxOth >= oth.primeexp.size()) {
                        /*
                         * exhausted the list in oth.primeexp; copy over the
                         * remaining 'this' if multiplying or lcm, discard if
                         * gcd.
                         */
                        if (type == 0 || type == 2) {
                            prod.primeexp.add(primeexp.elementAt(idxThis));
                            prod.primeexp.add(primeexp.elementAt(idxThis + 1));
                        }
                        idxThis += 2;
                    } else if (idxThis >= primeexp.size()) {
                        /*
                         * exhausted the list in primeexp; copy over the
                         * remaining 'oth'
                         */
                        if (type == 0 || type == 2) {
                            prod.primeexp.add(oth.primeexp.elementAt(idxOth));
                            prod.primeexp.add(oth.primeexp.elementAt(idxOth + 1));
                        }
                        idxOth += 2;
                    } else {
                        Integer p;
                        int ex;
                        switch (primeexp.elementAt(idxThis)
                                .compareTo(oth.primeexp.elementAt(idxOth))) {
                        case 0:
                            /* same prime bases p in both factors */
                            p = primeexp.elementAt(idxThis);
                            switch (type) {
                            case 0:
                                /* product means adding exponents */
                                ex = primeexp.elementAt(idxThis + 1).intValue()
                                        + oth.primeexp.elementAt(idxOth + 1).intValue();
                                break;
                            case 1:
                                /* gcd means minimum of exponents */
                                ex = Math.min(primeexp.elementAt(idxThis + 1).intValue(),
                                        oth.primeexp.elementAt(idxOth + 1).intValue());
                                break;
                            default:
                                /* lcm means maximum of exponents */
                                ex = Math.max(primeexp.elementAt(idxThis + 1).intValue(),
                                        oth.primeexp.elementAt(idxOth + 1).intValue());
                                break;
                            }
                            prod.primeexp.add(p);
                            prod.primeexp.add(new Integer(ex));
                            idxOth += 2;
                            idxThis += 2;
                            break;
                        case 1:
                            /*
                             * this prime base bigger than the other and taken
                             * later
                             */
                            if (type == 0 || type == 2) {
                                prod.primeexp.add(oth.primeexp.elementAt(idxOth));
                                prod.primeexp.add(oth.primeexp.elementAt(idxOth + 1));
                            }
                            idxOth += 2;
                            break;
                        default:
                            /*
                             * this prime base smaller than the other and taken
                             * now
                             */
                            if (type == 0 || type == 2) {
                                prod.primeexp.add(primeexp.elementAt(idxThis));
                                prod.primeexp.add(primeexp.elementAt(idxThis + 1));
                            }
                            idxThis += 2;
                        }
                    }
                }
            }
        }
        return prod;
    } /* Ifactor.multGcdLcm */


    public Ifactor divide(final Ifactor oth) {
        /*
         * todo: it'd probably be faster to cancel the gcd(this,oth) first in
         * the prime power representation, which would avoid a more strenous
         * factorization of the integer ratio
         */
        return new Ifactor(n.divide(oth.n));
    } /* Ifactor.divide */


    public Ifactor add(final BigInteger oth) {
        /*
         * avoid refactorization if oth is zero...
         */
        if (oth.compareTo(BigInteger.ZERO) != 0)
            return new Ifactor(n.add(oth));
        else
            return this;
    } /* Ifactor.add */


    public Ifactor pow(final int exponent) throws ArithmeticException {
        /*
         * three simple cases first
         */
        if (exponent < 0)
            throw new ArithmeticException(
                    "Cannot raise " + toString() + " to negative " + exponent);
        else if (exponent == 0)
            return new Ifactor(1);
        else if (exponent == 1)
            return this;

        /*
         * general case, the vector with the prime factor powers, which are
         * component-wise exponentiation of the individual prime factor powers.
         */
        Ifactor pows = new Ifactor(0);
        for (int i = 0; i < primeexp.size(); i += 2) {
            Integer p = primeexp.elementAt(i);
            int ex = primeexp.elementAt(i + 1).intValue();
            pows.primeexp.add(p);
            pows.primeexp.add(new Integer(ex * exponent));
        }
        return pows;
    } /* Ifactor.pow */


    public Rational root(final int r) throws ArithmeticException {
        if (r == 0)
            throw new ArithmeticException("Cannot pull zeroth root of " + toString());
        else if (r < 0) {
            /*
             * a^(-1/b)= 1/(a^(1/b))
             */
            final Rational invRoot = root(-r);
            return Rational.ONE.divide(invRoot);
        } else {
            BigInteger pows = BigInteger.ONE;
            for (int i = 0; i < primeexp.size(); i += 2) {
                /*
                 * all exponents must be multiples of r to succeed (that is, to
                 * stay in the range of rational results).
                 */
                int ex = primeexp.elementAt(i + 1).intValue();
                if (ex % r != 0)
                    throw new ArithmeticException("Cannot pull " + r + "th root of " + toString());

                pows.multiply(new BigInteger("" + primeexp.elementAt(i)).pow(ex / r));
            }
            /*
             * convert result to a Rational; unfortunately this will loose the
             * prime factorization
             */
            return new Rational(pows);
        }
    } /* Ifactor.root */


    public Vector<BigInteger> divisors() {
        /*
         * Recursive approach: the divisors of p1^e1*p2^e2*..*py^ey*pz^ez are
         * the divisors that don't contain the factor pz, and the the divisors
         * that contain any power of pz between 1 and up to ez multiplied by 1
         * or by a product that contains the factors p1..py.
         */
        Vector<BigInteger> d = new Vector<BigInteger>();
        if (n.compareTo(BigInteger.ZERO) == 0)
            return d;
        d.add(BigInteger.ONE);
        if (n.compareTo(BigInteger.ONE) > 0) {
            /* Computes sigmaIncopml(p1^e*p2^e2...*py^ey) */
            Ifactor dp = dropPrime();

            /* get ez */
            final int ez = primeexp.lastElement().intValue();

            Vector<BigInteger> partd = dp.divisors();

            /* obtain pz by lookup in the prime list */
            final BigInteger pz = new BigInteger(
                    primeexp.elementAt(primeexp.size() - 2).toString());

            /*
             * the output contains all products of the form partd[]*pz^ez, ez>0,
             * and with the exception of the 1, all these are appended.
             */
            for (int i = 1; i < partd.size(); i++)
                d.add(partd.elementAt(i));
            for (int e = 1; e <= ez; e++) {
                final BigInteger pzez = pz.pow(e);
                for (int i = 0; i < partd.size(); i++)
                    d.add(partd.elementAt(i).multiply(pzez));
            }
        }
        Collections.sort(d);
        return d;
    } /* Ifactor.divisors */


    public Ifactor sigma() {
        return sigma(1);
    } /* Ifactor.sigma */


    public Ifactor sigma(int k) {
        /*
         * the question is whether keeping a factorization is worth the effort
         * or whether one should simply multiply these to return a BigInteger...
         */
        if (n.compareTo(BigInteger.ONE) == 0)
            return ONE;
        else if (n.compareTo(BigInteger.ZERO) == 0)
            return ZERO;
        else {
            /*
             * multiplicative: sigma_k(p^e) = [p^(k*(e+1))-1]/[p^k-1]
             * sigma_0(p^e) = e+1.
             */
            Ifactor resul = Ifactor.ONE;
            for (int i = 0; i < primeexp.size(); i += 2) {
                int ex = primeexp.elementAt(i + 1).intValue();
                if (k == 0)
                    resul = resul.multiply(ex + 1);
                else {
                    Integer p = primeexp.elementAt(i);
                    BigInteger num = (new BigInteger(p.toString())).pow(k * (ex + 1))
                            .subtract(BigInteger.ONE);
                    BigInteger deno = (new BigInteger(p.toString())).pow(k)
                            .subtract(BigInteger.ONE);
                    /*
                     * This division is of course exact, no remainder The costly
                     * prime factorization is hidden here.
                     */
                    Ifactor f = new Ifactor(num.divide(deno));
                    resul = resul.multiply(f);
                }
            }
            return resul;
        }
    } /* Ifactor.sigma */


    public Ifactor dropPrime() {
        /*
         * the cases n==1 or n ==0
         */
        if (n.compareTo(BigInteger.ONE) <= 0)
            return this;

        /*
         * The cases n>1 Start empty. Copy all but the last factor over to the
         * result the vector with the new prime factor powers, which contain the
         * old prime factor powers up to but not including the last one.
         */
        Ifactor pows = new Ifactor(0);
        pows.n = BigInteger.ONE;
        for (int i = 0; i < primeexp.size() - 2; i += 2) {
            pows.primeexp.add(primeexp.elementAt(i));
            pows.primeexp.add(primeexp.elementAt(i + 1));
            BigInteger p = new BigInteger(primeexp.elementAt(i).toString());
            int ex = primeexp.elementAt(i + 1).intValue();
            pows.n = pows.n.multiply(p.pow(ex));
        }
        return pows;
    } /* Ifactor.dropPrime */


    public boolean issquare() {
        boolean resul = true;
        /*
         * check the exponents, located at the odd-indexed positions
         */
        for (int i = 1; i < primeexp.size(); i += 2) {
            if (primeexp.elementAt(i).intValue() % 2 != 0)
                return false;
        }
        return true;
    } /* Ifactor.issquare */


    public int bigomega() {
        int resul = 0;
        for (int i = 1; i < primeexp.size(); i += 2)
            resul += primeexp.elementAt(i).intValue();
        return (resul);
    } /* Ifactor.bigomega */


    public int omega() {
        return primeexp.size() / 2;
    } /* Ifactor.omega */


    public BigInteger core() {
        BigInteger resul = BigInteger.ONE;
        for (int i = 0; i < primeexp.size(); i += 2)
            if (primeexp.elementAt(i + 1).intValue() % 2 != 0)
                resul = resul.multiply(new BigInteger(primeexp.elementAt(i).toString()));
        return resul;
    } /* Ifactor.core */


    public int moebius() {
        if (n.compareTo(BigInteger.ONE) <= 0)
            return 1;
        /* accumulate number of different primes in k */
        int k = 1;
        for (int i = 0; i < primeexp.size(); i += 2) {
            final int e = primeexp.elementAt(i + 1).intValue();
            if (e > 1)
                return 0;
            else if (e == 1)
                /* accumulates (-1)^k */
                k *= -1;

        }
        return (k);
    } /* Ifactor.moebius */


    public Ifactor max(final Ifactor oth) {
        if (n.compareTo(oth.n) >= 0)
            return this;
        else
            return oth;
    } /* Ifactor.max */


    public Ifactor min(final Ifactor oth) {
        if (n.compareTo(oth.n) <= 0)
            return this;
        else
            return oth;
    } /* Ifactor.min */


    public static Ifactor max(final Vector<Ifactor> set) {
        Ifactor resul = set.elementAt(0);
        for (int i = 1; i < set.size(); i++)
            resul = resul.max(set.elementAt(i));
        return resul;
    } /* Ifactor.max */


    public static Ifactor min(final Vector<Ifactor> set) {
        Ifactor resul = set.elementAt(0);
        for (int i = 1; i < set.size(); i++)
            resul = resul.min(set.elementAt(i));
        return resul;
    } /* Ifactor.min */


    public int compareTo(final Ifactor oth) {
        return n.compareTo(oth.n);
    } /* compareTo */


    public String toString() {
        String resul = new String(n.toString() + ":");
        if (n.compareTo(BigInteger.ONE) == 0)
            resul += "1";
        else {
            boolean firstMul = true;
            for (int i = 0; i < primeexp.size(); i += 2) {
                if (!firstMul)
                    resul += "*";
                if (primeexp.elementAt(i + 1).intValue() > 1)
                    resul += primeexp.elementAt(i).toString() + "^"
                            + primeexp.elementAt(i + 1).toString();
                else
                    resul += primeexp.elementAt(i).toString();
                firstMul = false;
            }
        }
        return resul;
    } /* Ifactor.toString */


    public static void main(String[] args) throws Exception {
        BigInteger n = new BigInteger(args[0]);
        System.out.println(new Ifactor(n));
    } /* Ifactor.main */
} /* Ifactor */
