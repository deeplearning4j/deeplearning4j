package org.nd4j.autodiff.bigdecimal;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.MathContext;
import java.math.RoundingMode;


public class Rational implements Cloneable, Comparable<Rational> {

    BigInteger a;


    BigInteger b;


    static public BigInteger MAX_INT = new BigInteger("2147483647");
    static public BigInteger MIN_INT = new BigInteger("-2147483648");


    static Rational ONE = new Rational(1, 1);

    static public Rational ZERO = new Rational();


    static public Rational HALF = new Rational(1, 2);


    public Rational() {
        a = BigInteger.ZERO;
        b = BigInteger.ONE;
    }


    public Rational(BigInteger a, BigInteger b) {
        this.a = a;
        this.b = b;
        normalize();
    }


    public Rational(BigInteger a) {
        this.a = a;
        b = new BigInteger("1");
    }


    public Rational(int a, int b) {
        this(new BigInteger("" + a), new BigInteger("" + b));
    }


    public Rational(int n) {
        this(n, 1);
    }


    public Rational(String str) throws NumberFormatException {
        this(str, 10);
    }


    public Rational(String str, int radix) throws NumberFormatException {
        int hasslah = str.indexOf("/");
        if (hasslah == -1) {
            a = new BigInteger(str, radix);
            b = new BigInteger("1", radix);
            /* no normalization necessary here */
        } else {
            /*
             * create numerator and denominator separately
             */
            a = new BigInteger(str.substring(0, hasslah), radix);
            b = new BigInteger(str.substring(hasslah + 1), radix);
            normalize();
        }
    }


    public Rational clone() {
        /*
         * protected access means this does not work return new
         * Rational(a.clone(), b.clone()) ;
         */
        BigInteger aclon = new BigInteger("" + a);
        BigInteger bclon = new BigInteger("" + b);
        return new Rational(aclon, bclon);
    } /* Rational.clone */


    public Rational multiply(final Rational val) {
        BigInteger num = a.multiply(val.a);
        BigInteger deno = b.multiply(val.b);
        /*
         * Normalization to an coprime format will be done inside the ctor() and
         * is not duplicated here.
         */
        return (new Rational(num, deno));
    } /* Rational.multiply */


    public Rational multiply(final BigInteger val) {
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (multiply(val2));
    } /* Rational.multiply */


    public Rational multiply(final int val) {
        BigInteger tmp = new BigInteger("" + val);
        return multiply(tmp);
    } /* Rational.multiply */


    public Rational pow(int exponent) {
        if (exponent == 0)
            return new Rational(1, 1);

        BigInteger num = a.pow(Math.abs(exponent));
        BigInteger deno = b.pow(Math.abs(exponent));
        if (exponent > 0)
            return (new Rational(num, deno));
        else
            return (new Rational(deno, num));
    } /* Rational.pow */


    public Rational pow(BigInteger exponent) throws NumberFormatException {
        /* test for overflow */
        if (exponent.compareTo(MAX_INT) == 1)
            throw new NumberFormatException("Exponent " + exponent.toString() + " too large.");
        if (exponent.compareTo(MIN_INT) == -1)
            throw new NumberFormatException("Exponent " + exponent.toString() + " too small.");

        /* promote to the simpler interface above */
        return pow(exponent.intValue());
    } /* Rational.pow */


    public Rational root(BigInteger r) throws NumberFormatException {
        /* test for overflow */
        if (r.compareTo(MAX_INT) == 1)
            throw new NumberFormatException("Root " + r.toString() + " too large.");
        if (r.compareTo(MIN_INT) == -1)
            throw new NumberFormatException("Root " + r.toString() + " too small.");

        int rthroot = r.intValue();
        /* cannot pull root of a negative value with even-valued root */
        if (compareTo(ZERO) == -1 && (rthroot % 2) == 0)
            throw new NumberFormatException(
                    "Negative basis " + toString() + " with odd root " + r.toString());

        /*
         * extract a sign such that we calculate |n|^(1/r), still r carrying any
         * sign
         */
        final boolean flipsign = (compareTo(ZERO) == -1 && (rthroot % 2) != 0) ? true : false;

        /*
         * delegate the main work to ifactor#root()
         */
        Ifactor num = new Ifactor(a.abs());
        Ifactor deno = new Ifactor(b);
        final Rational resul = num.root(rthroot).divide(deno.root(rthroot));
        if (flipsign)
            return resul.negate();
        else
            return resul;
    } /* Rational.root */


    public Rational pow(Rational exponent) throws NumberFormatException {
        if (exponent.a.compareTo(BigInteger.ZERO) == 0)
            return new Rational(1, 1);

        /*
         * calculate (a/b)^(exponent.a/exponent.b) as
         * ((a/b)^exponent.a)^(1/exponent.b) = tmp^(1/exponent.b)
         */
        Rational tmp = pow(exponent.a);
        return tmp.root(exponent.b);
    } /* Rational.pow */


    public Rational divide(final Rational val) {
        if (val.compareTo(Rational.ZERO) == 0)
            throw new ArithmeticException("Dividing " + toString() + " through zero.");
        BigInteger num = a.multiply(val.b);
        BigInteger deno = b.multiply(val.a);
        /*
         * Reduction to a coprime format is done inside the ctor, and not
         * repeated here.
         */
        return (new Rational(num, deno));
    } /* Rational.divide */


    public Rational divide(BigInteger val) {
        if (val.compareTo(BigInteger.ZERO) == 0)
            throw new ArithmeticException("Dividing " + toString() + " through zero.");
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (divide(val2));
    } /* Rational.divide */


    public Rational divide(int val) {
        if (val == 0)
            throw new ArithmeticException("Dividing " + toString() + " through zero.");
        Rational val2 = new Rational(val, 1);
        return (divide(val2));
    } /* Rational.divide */


    public Rational add(Rational val) {
        BigInteger num = a.multiply(val.b).add(b.multiply(val.a));
        BigInteger deno = b.multiply(val.b);
        return (new Rational(num, deno));
    } /* Rational.add */


    public Rational add(BigInteger val) {
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (add(val2));
    } /* Rational.add */


    public Rational add(int val) {
        BigInteger val2 = a.add(b.multiply(new BigInteger("" + val)));
        return new Rational(val2, b);
    } /* Rational.add */


    public Rational negate() {
        return (new Rational(a.negate(), b));
    } /* Rational.negate */


    public Rational subtract(Rational val) {
        Rational val2 = val.negate();
        return (add(val2));
    } /* Rational.subtract */


    public Rational subtract(BigInteger val) {
        Rational val2 = new Rational(val, BigInteger.ONE);
        return (subtract(val2));
    } /* Rational.subtract */


    public Rational subtract(int val) {
        Rational val2 = new Rational(val, 1);
        return (subtract(val2));
    } /* Rational.subtract */


    public static Rational binomial(Rational n, BigInteger m) {
        if (m.compareTo(BigInteger.ZERO) == 0)
            return Rational.ONE;
        Rational bin = n;
        for (BigInteger i = new BigInteger("2"); i.compareTo(m) != 1; i = i.add(BigInteger.ONE)) {
            bin = bin.multiply(n.subtract(i.subtract(BigInteger.ONE))).divide(i);
        }
        return bin;
    } /* Rational.binomial */


    public static Rational binomial(Rational n, int m) {
        if (m == 0)
            return Rational.ONE;
        Rational bin = n;
        for (int i = 2; i <= m; i++) {
            bin = bin.multiply(n.subtract(i - 1)).divide(i);
        }
        return bin;
    } /* Rational.binomial */


    public static Rational hankelSymb(Rational n, int k) {
        if (k == 0)
            return Rational.ONE;
        else if (k < 0)
            throw new ArithmeticException("Negative parameter " + k);
        Rational nkhalf = n.subtract(k).add(Rational.HALF);
        nkhalf = nkhalf.Pochhammer(2 * k);
        Factorial f = new Factorial();
        return nkhalf.divide(f.at(k));
    } /* Rational.binomial */


    public BigInteger numer() {
        return a;
    }


    public BigInteger denom() {
        return b;
    }


    public Rational abs() {
        return (new Rational(a.abs(), b.abs()));
    }


    public BigInteger floor() {
        /*
         * is already integer: return the numerator
         */
        if (b.compareTo(BigInteger.ONE) == 0)
            return a;
        else if (a.compareTo(BigInteger.ZERO) > 0)
            return a.divide(b);
        else
            return a.divide(b).subtract(BigInteger.ONE);
    } /* Rational.floor */


    public BigInteger ceil() {
        /*
         * is already integer: return the numerator
         */
        if (b.compareTo(BigInteger.ONE) == 0)
            return a;
        else if (a.compareTo(BigInteger.ZERO) > 0)
            return a.divide(b).add(BigInteger.ONE);
        else
            return a.divide(b);
    } /* Rational.ceil */


    public BigInteger trunc() {
        /*
         * is already integer: return the numerator
         */
        if (b.compareTo(BigInteger.ONE) == 0)
            return a;
        else
            return a.divide(b);
    } /* Rational.trunc */


    public int compareTo(final Rational val) {
        /*
         * Since we have always kept the denominators positive, simple
         * cross-multiplying works without changing the sign.
         */
        final BigInteger left = a.multiply(val.b);
        final BigInteger right = val.a.multiply(b);
        return left.compareTo(right);
    } /* Rational.compareTo */


    public int compareTo(final BigInteger val) {
        final Rational val2 = new Rational(val, BigInteger.ONE);
        return (compareTo(val2));
    } /* Rational.compareTo */


    public String toString() {
        if (b.compareTo(BigInteger.ONE) != 0)
            return (a.toString() + "/" + b.toString());
        else
            return a.toString();
    } /* Rational.toString */


    public double doubleValue() {
        /*
         * To meet the risk of individual overflows of the exponents of a
         * separate invocation a.doubleValue() or b.doubleValue(), we divide
         * first in a BigDecimal environment and convert the result.
         */
        BigDecimal adivb = (new BigDecimal(a)).divide(new BigDecimal(b), MathContext.DECIMAL128);
        return adivb.doubleValue();
    } /* Rational.doubleValue */


    public float floatValue() {
        BigDecimal adivb = (new BigDecimal(a)).divide(new BigDecimal(b), MathContext.DECIMAL128);
        return adivb.floatValue();
    } /* Rational.floatValue */


    public BigDecimal BigDecimalValue(MathContext mc) {
        /*
         * numerator and denominator individually rephrased
         */
        BigDecimal n = new BigDecimal(a);
        BigDecimal d = new BigDecimal(b);
        /*
         * the problem with n.divide(d,mc) is that the apparent precision might
         * be smaller than what is set by mc if the value has a precise
         * truncated representation. 1/4 will appear as 0.25, independent of mc
         */
        return BigDecimalMath.scalePrec(n.divide(d, mc), mc);
    } /* Rational.BigDecimalValue */


    public String toFString(int digits) {
        if (b.compareTo(BigInteger.ONE) != 0) {
            MathContext mc = new MathContext(digits, RoundingMode.DOWN);
            BigDecimal f = (new BigDecimal(a)).divide(new BigDecimal(b), mc);
            return (f.toString());
        } else
            return a.toString();
    } /* Rational.toFString */


    public Rational max(final Rational val) {
        if (compareTo(val) > 0)
            return this;
        else
            return val;
    } /* Rational.max */


    public Rational min(final Rational val) {
        if (compareTo(val) < 0)
            return this;
        else
            return val;
    } /* Rational.min */


    public Rational Pochhammer(final BigInteger n) {
        if (n.compareTo(BigInteger.ZERO) < 0)
            return null;
        else if (n.compareTo(BigInteger.ZERO) == 0)
            return Rational.ONE;
        else {
            /*
             * initialize results with the current value
             */
            Rational res = new Rational(a, b);
            BigInteger i = BigInteger.ONE;
            for (; i.compareTo(n) < 0; i = i.add(BigInteger.ONE))
                res = res.multiply(add(i));
            return res;
        }
    } /* Rational.pochhammer */


    public Rational Pochhammer(int n) {
        return Pochhammer(new BigInteger("" + n));
    } /* Rational.pochhammer */


    public boolean isBigInteger() {
        return (b.abs().compareTo(BigInteger.ONE) == 0);
    } /* Rational.isBigInteger */


    public boolean isInteger() {
        if (!isBigInteger())
            return false;
        return (a.compareTo(MAX_INT) <= 0 && a.compareTo(MIN_INT) >= 0);
    } /* Rational.isInteger */


    int intValue() {
        if (!isInteger())
            throw new NumberFormatException("cannot convert " + toString() + " to integer.");
        return a.intValue();
    }


    BigInteger BigIntegerValue() {
        if (!isBigInteger())
            throw new NumberFormatException("cannot convert " + toString() + " to BigInteger.");
        return a;
    }


    public boolean isIntegerFrac() {
        return (a.compareTo(MAX_INT) <= 0 && a.compareTo(MIN_INT) >= 0 && b.compareTo(MAX_INT) <= 0
                && b.compareTo(MIN_INT) >= 0);
    } /* Rational.isIntegerFrac */


    public int signum() {
        return (b.signum() * a.signum());
    } /* Rational.signum */


    static public BigInteger lcmDenom(final Rational[] vals) {
        BigInteger l = BigInteger.ONE;
        for (int v = 0; v < vals.length; v++)
            l = BigIntegerMath.lcm(l, vals[v].b);
        return l;
    } /* Rational.lcmDenom */


    protected void normalize() {
        /*
         * compute greatest common divisor of numerator and denominator
         */
        final BigInteger g = a.gcd(b);
        if (g.compareTo(BigInteger.ONE) > 0) {
            a = a.divide(g);
            b = b.divide(g);
        }
        if (b.compareTo(BigInteger.ZERO) == -1) {
            a = a.negate();
            b = b.negate();
        }
    } /* Rational.normalize */
} /* Rational */
