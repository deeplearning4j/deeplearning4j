package org.nd4j.autodiff.doubledouble;

import java.io.Serializable;

import com.google.common.base.Objects;
import com.google.common.math.DoubleMath;

public strictfp class DoubleDouble implements Serializable, Comparable<DoubleDouble>, Cloneable {


    private static final DoubleDoubleCache<DoubleDouble> EXP_CACHE = new DoubleDoubleCache<>();
    private static final DoubleDoubleCache<CacheMap<Integer, DoubleDouble>> POW_INTEGER_CACHE = new DoubleDoubleCache<>();
    private static final int POW_INT_CACHE_SIZE_LIMIT = 100000;
    private static final DoubleDoubleCache<DoubleDoubleCache<DoubleDouble>> POW_DOUBLE_DOUBLE_CACHE = new DoubleDoubleCache<>();

    private static final long serialVersionUID = 1L;

    public static final DoubleDouble ZERO = fromOneDouble(0.0);

    public static final DoubleDouble HALF = fromOneDouble(0.5);

    public static final DoubleDouble PI = fromTwoDouble(3.141592653589793116e+00,
            1.224646799147353207e-16);

    public static final DoubleDouble TWO_PI = fromTwoDouble(6.283185307179586232e+00,
            2.449293598294706414e-16);

    public static final DoubleDouble PI_2 = fromTwoDouble(1.570796326794896558e+00,
            6.123233995736766036e-17);

    public static final DoubleDouble E = fromTwoDouble(2.718281828459045091e+00,
            1.445646891729250158e-16);

    public static final DoubleDouble NaN = fromTwoDouble(Double.NaN, Double.NaN);

    public static final DoubleDouble POSITIVE_INFINITY = fromTwoDouble(Double.POSITIVE_INFINITY,
            Double.POSITIVE_INFINITY);

    public static final DoubleDouble NEGATIVE_INFINITY = fromTwoDouble(Double.NEGATIVE_INFINITY,
            Double.NEGATIVE_INFINITY);

    public static final double EPS = 1.23259516440783e-32; /* = 2^-106 */
    private static final double SPLIT = 134217729.0D; // 2^27+1, for IEEE double

    public static final DoubleDouble ONE = fromOneDouble(1.0);
    public static final DoubleDouble MINUS_ONE = fromOneDouble(-1.0);
    public static final DoubleDouble TWO = fromOneDouble(2.0);
    private static final DoubleDouble TEN = fromOneDouble(10.0);
    private static final DoubleDouble LOG_TEN = TEN.log();

    public static DoubleDouble fromString(String str) {
        int i = 0;
        final int strlen = str.length();

        // skip leading whitespace
        while (Character.isWhitespace(str.charAt(i))) {
            i++;
        }

        // check for sign
        boolean isNegative = false;
        if (i < strlen) {
            final char signCh = str.charAt(i);
            if (signCh == '-' || signCh == '+') {
                i++;
                if (signCh == '-') {
                    isNegative = true;
                }
            }
        }

        // scan all digits and accumulate into an integral value
        // Keep track of the location of the decimal point (if any) to allow
        // scaling later
        DoubleDouble val = ZERO;

        int numDigits = 0;
        int numBeforeDec = 0;
        int exp = 0;
        while (true) {
            if (i >= strlen) {
                break;
            }
            final char ch = str.charAt(i);
            i++;
            if (Character.isDigit(ch)) {
                final double d = ch - '0';
                val = val.multiply(TEN);
                // MD: need to optimize this
                val = val.add(fromOneDouble(d));
                numDigits++;
                continue;
            }
            if (ch == '.') {
                numBeforeDec = numDigits;
                continue;
            }
            if (ch == 'e' || ch == 'E') {
                final String expStr = str.substring(i);
                // this should catch any format problems with the exponent
                try {
                    exp = Integer.parseInt(expStr);
                } catch (final NumberFormatException ex) {
                    throw new NumberFormatException(
                            "Invalid exponent " + expStr + " in string " + str);
                }
                break;
            }
            throw new NumberFormatException(
                    "Unexpected character '" + ch + "' at position " + i + " in string " + str);
        }

        if (numBeforeDec == 0) {
            numBeforeDec = numDigits;
        }
        DoubleDouble val2;

        // scale the number correctly
        final int numDecPlaces = numDigits - numBeforeDec - exp;
        if (numDecPlaces == 0) {
            val2 = val;
        } else if (numDecPlaces > 0) {
            final DoubleDouble scale = TEN.pow(numDecPlaces);
            val2 = val.divide(scale);
        } else {
            final DoubleDouble scale = TEN.pow(-numDecPlaces);
            val2 = val.multiply(scale);
        }
        // apply leading sign, if any
        if (isNegative) {
            return val2.negate();
        }
        return val2;

    }

    private final boolean isNaN;
    private final double hi;
    private final double lo;

    private DoubleDouble(double hi, double lo) {
        this.isNaN = Double.isNaN(hi);
        this.hi = hi;
        this.lo = lo;
    }

    public static DoubleDouble fromTwoDouble(double hi, double lo) {
        return new DoubleDouble(hi, lo);
    }

    public static DoubleDouble fromOneDouble(double x) {
        return fromTwoDouble(x, 0.0);
    }

    public static DoubleDouble fromDoubleDouble(DoubleDouble dd) {
        return fromTwoDouble(dd.hi, dd.lo);
    }

    public DoubleDouble abs() {
        if (isNaN) {
            return NaN;
        }
        if (isNegative()) {
            return negate();
        }
        return fromDoubleDouble(this);
    }

    public DoubleDouble acos() {
        if (isNaN) {
            return NaN;
        }
        if ((this.abs()).gt(ONE)) {
            return NaN;
        }
        final DoubleDouble s = PI_2.subtract(this.asin());
        return s;
    }

    /*
     * // experimental private DoubleDouble innerAdd(double yhi, double ylo) {
     * double H, h, T, t, S, s, e, f; S = hi + yhi; T = lo + ylo; e = S - hi; f
     * = T - lo; s = S-e; t = T-f; s = (yhi-e)+(hi-s); t = (ylo-f)+(lo-t); e =
     * s+T; H = S+e; h = e+(S-H); e = t+h;
     *
     * double zhi = H + e; double zlo = e + (H - zhi); hi = zhi; lo = zlo;
     *
     * return this; }
     */

    public DoubleDouble asin() {
        // Return the arcsine of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }
        if ((this.abs()).gt(ONE)) {
            return NaN;
        }
        if (this.equals(ONE)) {
            return PI_2;
        }
        if (this.equals(MINUS_ONE)) {
            return PI_2.negate();
        }
        final DoubleDouble square = multiply(this);
        DoubleDouble s = fromDoubleDouble(this);
        DoubleDouble sOld;
        DoubleDouble t = fromDoubleDouble(this);
        DoubleDouble w;
        double n = 1.0;
        double numn;
        double denomn;
        do {
            n += 2.0;
            numn = n - 2.0;
            denomn = n - 1.0;
            t = t.divide(fromOneDouble(denomn));
            t = t.multiply(fromOneDouble(numn));
            t = t.multiply(square);
            w = t.divide(fromOneDouble(n));
            sOld = s;
            s = s.add(w);
        } while (s.ne(sOld));
        return s;
    }

    public DoubleDouble atan() {
        if (isNaN) {
            return NaN;
        }
        DoubleDouble s;
        DoubleDouble sOld;
        if (this.equals(ONE)) {
            s = PI_2.divide(TWO);
        } else if (this.equals(MINUS_ONE)) {
            s = PI_2.divide(fromOneDouble(-2.0));
        } else if (this.abs().lt(ONE)) {
            final DoubleDouble msquare = (this.multiply(this)).negate();
            s = fromDoubleDouble(this);

            DoubleDouble t = fromDoubleDouble(this);
            DoubleDouble w;
            double n = 1.0;
            do {
                n += 2.0;
                t = t.multiply(msquare);
                w = t.divide(fromOneDouble(n));
                sOld = s;
                s = s.add(w);
            } while (s.ne(sOld));
        } else {
            final DoubleDouble msquare = (this.multiply(this)).negate();
            s = this.reciprocal().negate();
            DoubleDouble t = fromDoubleDouble(s);
            DoubleDouble w;
            double n = 1.0;
            do {
                n += 2.0;
                t = t.divide(msquare);
                w = t.divide(fromOneDouble(n));
                sOld = s;
                s = s.add(w);
            } while (s.ne(sOld));
            if (isPositive()) {
                s = s.add(PI_2);
            } else {
                s = s.subtract(PI_2);
            }
        }
        return s;
    }

    public DoubleDouble BernoulliA(int n) {
        // For PI/2 < x < PI, sum from k = 1 to infinity of
        // ((-1)**(k-1))*sin(kx)/(k**2) =
        // x*ln(2) - sum from k = 1 to infinity of
        // ((-1)**(k-1))*(2**(2k) - 1)*B2k*(x**(2*k+1))/(((2*k)!)*(2*k)*(2*k+1))
        // Compute the DoubleDouble Bernoulli number Bn
        // Ported from subroutine BERNOA in
        // Computation of Special Functions by Shanjie Zhang and Jianming Jin
        // I thought of creating a version with all the Bernoulli numbers from
        // B0 to Bn-1 passed in as an input to calculate Bn. However, according
        // Zhang and Jin using the correct zero values for B3, B5, B7 actually
        // gives
        // a much worse result than using the incorrect intermediate B3, B5, B7
        // values caluclated by this algorithm.
        int m;
        int k;
        int j;
        DoubleDouble s;
        DoubleDouble r;
        DoubleDouble temp;
        if (n < 0) {
            return NaN;
        } else if ((n >= 3) && (((n - 1) % 2) == 0)) {
            // B2*n+1 = 0 for n = 1,2,3
            return ZERO;
        }
        final DoubleDouble BN[] = new DoubleDouble[n + 1];
        BN[0] = ONE;
        if (n == 0) {
            return BN[0];
        }
        BN[1] = fromOneDouble(-0.5);
        if (n == 1) {
            return BN[1];
        }
        for (m = 2; m <= n; m++) {
            s = (fromOneDouble(m)).add(ONE);
            s = s.reciprocal();
            s = (HALF).subtract(s);
            for (k = 2; k <= m - 1; k++) {
                r = ONE;
                for (j = 2; j <= k; j++) {
                    temp = (fromOneDouble(j)).add(fromOneDouble(m));
                    temp = temp.subtract(fromOneDouble(k));
                    temp = temp.divide(fromOneDouble(j));
                    r = r.multiply(temp);
                }
                temp = r.multiply(BN[k]);
                s = s.subtract(temp);
            }
            BN[m] = s;
        }
        return BN[n];
    }

    public DoubleDouble BernoulliB(int n) {
        // B2n = ((-1)**(n-1))*2*((2*n)!)*(1 + 1/(2**(2*n)) + 1/(3**(2*n)) +
        // ...)/((2*PI)**(2*n))
        // = ((-1)**(n-1))*2*(1 + 1/(2**(2*n)) + 1/(3**(2*n)) + ...) * product
        // from m = 1 to 2n of m/(2*PI)
        // for n = 1, 2, 3, ...
        // Compute the DoubleDouble Bernoulli number Bn
        // More efficient than BernoulliA
        // Ported from subroutine BERNOB in
        // Computation of Special Functions by Shanjie Zhang and Jianming Jin
        int k;
        DoubleDouble r1;
        DoubleDouble twoPISqr;
        DoubleDouble r2;
        DoubleDouble s;
        DoubleDouble sOld;
        DoubleDouble temp;
        if (n < 0) {
            return NaN;
        } else if ((n >= 3) && (((n - 1) % 2) == 0)) {
            // B2*n+1 = 0 for n = 1,2,3
            return ZERO;
        }
        DoubleDouble bn[] = new DoubleDouble[n + 1];
        bn[0] = ONE;
        if (n == 0) {
            return bn[0];
        }
        bn[1] = fromOneDouble(-0.5);
        if (n == 1) {
            return bn[1];
        }
        bn[2] = (ONE).divide(fromOneDouble(6.0));
        if (n == 2) {
            return bn[2];
        }
        r1 = ((ONE).divide(PI)).sqr();
        twoPISqr = TWO_PI.multiply(TWO_PI);
        for (int m = 4; m <= n; m += 2) {
            temp = (fromOneDouble(m)).divide(twoPISqr);
            temp = (fromOneDouble(m - 1d)).multiply(temp);
            r1 = (r1.multiply(temp)).negate();
            r2 = ONE;
            s = ONE;
            k = 2;
            do {
                sOld = s;
                s = (ONE).divide(fromOneDouble(k++));
                s = s.pow(m);
                r2 = r2.add(s);
            } while (s.ne(sOld));
            bn[m] = r1.multiply(r2);
        }
        return bn[n];
    }

    /*
     *
     * // experimental public DoubleDouble selfDivide(DoubleDouble y) { double
     * hc, tc, hy, ty, C, c, U, u; C = hi/y.hi; c = SPLIT*C; hc =c-C; u =
     * SPLIT*y.hi; hc = c-hc; tc = C-hc; hy = u-y.hi; U = C * y.hi; hy = u-hy;
     * ty = y.hi-hy; u = (((hc*hy-U)+hc*ty)+tc*hy)+tc*ty; c =
     * ((((hi-U)-u)+lo)-C*y.lo)/y.hi; u = C+c;
     *
     * hi = u; lo = (C-u)+c; return this; }
     */

    public DoubleDouble ceil() {
        if (isNaN) {
            return NaN;
        }
        final double fhi = Math.ceil(hi);
        double flo = 0.0;
        // Hi is already integral. Ceil the low word
        if (fhi == hi) {
            flo = Math.ceil(lo);
            // do we need to renormalize here?
        }
        return fromTwoDouble(fhi, flo);
    }

    public DoubleDouble round() {
        return this.add(HALF).floor();
    }

    public DoubleDouble Ci() {
        final DoubleDouble x = fromDoubleDouble(this);
        final DoubleDouble Ci = ZERO;
        final DoubleDouble Si = ZERO;
        cisia(x, Ci, Si);
        return Ci;
    }

    public void cisia(DoubleDouble x, DoubleDouble Ci, DoubleDouble Si) {
        // Purpose: Compute cosine and sine integrals Si(x) and
        // Ci(x) (x >= 0)
        // Input x: Argument of Ci(x) and Si(x)
        // Output: Ci(x), Si(x)
        final DoubleDouble bj[] = new DoubleDouble[101];
        // Euler's constant
        final DoubleDouble el = fromOneDouble(.57721566490153286060651209008240243104215933593992);
        DoubleDouble x2;
        DoubleDouble xr;
        int k;
        int m;
        DoubleDouble xa1;
        DoubleDouble xa0;
        DoubleDouble xa;
        DoubleDouble xs;
        DoubleDouble xg1;
        DoubleDouble xg2;
        DoubleDouble xf;
        DoubleDouble xg;
        int i1;
        int i2;
        DoubleDouble var1;
        DoubleDouble var2;

        x2 = x.multiply(x);
        if (x.le(fromOneDouble(16.0))) {
            xr = (fromOneDouble(-0.25)).multiply(x2);
            Ci = (el.add(x.log())).add(xr);
            for (k = 2; k <= 40; k++) {
                xr = ((((fromOneDouble(-0.5)).multiply(xr)).multiply(fromOneDouble(k - 1))).divide(
                        fromOneDouble(k * k * (2 * k - 1)))).multiply(x2);
                Ci = Ci.add(xr);
                if ((xr.abs()).lt((Ci.abs()).multiply(fromOneDouble(EPS)))) {
                    break;
                }
            }
            xr = fromDoubleDouble(x);
            Si = fromDoubleDouble(x);
            for (k = 1; k <= 40; k++) {
                xr = (((((fromOneDouble(-0.5)).multiply(xr)).multiply(
                        fromOneDouble(2 * k - 1))).divide(fromOneDouble(k))).divide(
                        fromOneDouble(4 * k * k + 4 * k + 1))).multiply(x2);
                Si = Si.add(xr);
                if ((xr.abs()).lt((Si.abs()).multiply(fromOneDouble(EPS)))) {
                    return;
                }
            }
        } // else if x <= 16
        else if (x.le(fromOneDouble(32.0))) {
            m = (((fromOneDouble(47.2)).add((fromOneDouble(0.82)).multiply(x))).trunc()).intValue();
            xa1 = ZERO;
            xa0 = fromOneDouble(1.0E-100);
            for (k = m; k >= 1; k--) {
                xa = ((((fromOneDouble(4.0)).multiply(fromOneDouble(k))).multiply(xa0)).divide(
                        x)).subtract(xa1);
                bj[k - 1] = fromDoubleDouble(xa);
                xa1 = fromDoubleDouble(xa0);
                xa0 = fromDoubleDouble(xa);
            } // for (k = m; k >= 1; k--)
            xs = fromDoubleDouble(bj[0]);
            for (k = 2; k <= m; k += 2) {
                xs = xs.add((TWO).multiply(bj[k]));
            }
            bj[0] = bj[0].divide(xs);
            for (k = 1; k <= m; k++) {
                bj[k] = bj[k].divide(xs);
            }
            xr = ONE;
            xg1 = fromDoubleDouble(bj[0]);
            for (k = 1; k <= m; k++) {
                i1 = (2 * k - 1) * (2 * k - 1);
                var1 = fromOneDouble(i1);
                i2 = k * (2 * k + 1) * (2 * k + 1);
                var2 = fromOneDouble(i2);
                xr = ((((fromOneDouble(0.25)).multiply(xr)).multiply(var1)).divide(var2)).multiply(
                        x);
                xg1 = xg1.add(bj[k].multiply(xr));
            }
            xr = ONE;
            xg2 = fromDoubleDouble(bj[0]);
            for (k = 1; k <= m; k++) {
                i1 = (2 * k - 3) * (2 * k - 3);
                var1 = fromOneDouble(i1);
                i2 = k * (2 * k - 1) * (2 * k - 1);
                var2 = fromOneDouble(i2);
                xr = ((((fromOneDouble(0.25)).multiply(xr)).multiply(var1)).divide(var2)).multiply(
                        x);
                xg2 = xg2.add(bj[k].multiply(xr));
            }
        } // else if x <= 32
        else {
            xr = ONE;
            xf = ONE;
            for (k = 1; k <= 9; k++) {
                i1 = k * (2 * k - 1);
                var1 = fromOneDouble(i1);
                xr = (((fromOneDouble(-2.0)).multiply(xr)).multiply(var1)).divide(x2);
                xf = xf.add(xr);
            }
            xr = x.reciprocal();
            xg = fromDoubleDouble(xr);
            for (k = 1; k <= 8; k++) {
                i1 = (2 * k + 1) * k;
                var1 = fromOneDouble(i1);
                xr = (((fromOneDouble(-2.0)).multiply(xr)).multiply(var1)).divide(x2);
                xg = xg.add(xr);
            }
        }
    }

    @Override
    public int compareTo(DoubleDouble other) {

        if (hi < other.hi) {
            return -1;
        }
        if (hi > other.hi) {
            return 1;
        }
        if (lo < other.lo) {
            return -1;
        }
        if (lo > other.lo) {
            return 1;
        }
        return 0;
    }

    public DoubleDouble cos() {
        boolean negate = false;
        // Return the cosine of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }
        DoubleDouble twoPIFullTimes;
        DoubleDouble twoPIremainder;
        if ((this.abs()).gt(TWO_PI)) {
            twoPIFullTimes = (this.divide(TWO_PI)).trunc();
            twoPIremainder = this.subtract(TWO_PI.multiply(twoPIFullTimes));
        } else {
            twoPIremainder = this;
        }
        if (twoPIremainder.gt(PI)) {
            twoPIremainder = twoPIremainder.subtract(PI);
            negate = true;
        } else if (twoPIremainder.lt(PI.negate())) {
            twoPIremainder = twoPIremainder.add(PI);
            negate = true;
        }
        final DoubleDouble msquare = (twoPIremainder.multiply(twoPIremainder)).negate();
        DoubleDouble s = ONE;
        DoubleDouble sOld;
        DoubleDouble t = ONE;
        double n = 0.0;
        do {
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            t = t.multiply(msquare);
            sOld = s;
            s = s.add(t);
        } while (s.ne(sOld));
        if (negate) {
            s = s.negate();
        }
        return s;
    }

    public DoubleDouble cosh() {
        // Return the cosh of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }
        final DoubleDouble square = this.multiply(this);
        DoubleDouble s = ONE;
        DoubleDouble sOld;
        DoubleDouble t = ONE;
        double n = 0.0;
        do {
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            t = t.multiply(square);
            sOld = s;
            s = s.add(t);
        } while (s.ne(sOld));
        return s;
    }

    public DoubleDouble divide(DoubleDouble y) {
        double hc, tc, hy, ty, C, c, U, u;
        C = hi / y.hi;
        c = SPLIT * C;
        hc = c - C;
        u = SPLIT * y.hi;
        hc = c - hc;
        tc = C - hc;
        hy = u - y.hi;
        U = C * y.hi;
        hy = u - hy;
        ty = y.hi - hy;
        u = (((hc * hy - U) + hc * ty) + tc * hy) + tc * ty;
        c = ((((hi - U) - u) + lo) - C * y.lo) / y.hi;
        u = C + c;

        final double zhi = u;
        final double zlo = (C - u) + c;
        return fromTwoDouble(zhi, zlo);
    }

    public double doubleValue() {
        return hi + lo;
    }

    public String dump() {
        return "DD<" + hi + ", " + lo + ">";
    }

    private DoubleDouble innerExp() {
        boolean invert = false;
        // Return the exponential of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }

        if (doubleValue() > 690) {
            return POSITIVE_INFINITY;
        }

        if (doubleValue() < -690) {
            return ZERO;
        }

        DoubleDouble x = this;
        if (x.lt(ZERO)) {
            // Much greater precision if all numbers in the series have the same
            // sign.
            x = x.negate();
            invert = true;
        }
        DoubleDouble s = ONE.add(x);
        DoubleDouble sOld;
        DoubleDouble t = fromDoubleDouble(x);
        double n = 1.0;

        do {
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            t = t.multiply(x);
            sOld = s;
            s = s.add(t);
        } while (s.ne(sOld));
        if (invert) {
            s = s.reciprocal();
        }
        return s;

    }

    public DoubleDouble factorial(int fac) {
        DoubleDouble prod;
        if (fac < 0) {
            return NaN;
        }
        if ((fac >= 0) && (fac <= 1)) {
            return ONE;
        }
        prod = fromOneDouble(fac--);
        while (fac > 1) {
            prod = prod.multiply(fromOneDouble(fac--));
        }
        return prod;
    }

    public DoubleDouble floor() {
        if (isNaN) {
            return NaN;
        }
        final double fhi = Math.floor(hi);
        double flo = 0.0;
        // Hi is already integral. Floor the low word
        if (fhi == hi) {
            flo = Math.floor(lo);
        }
        // do we need to renormalize here?
        return fromTwoDouble(fhi, flo);
    }

    public boolean ge(DoubleDouble y) {
        return (hi > y.hi) || (hi == y.hi && lo >= y.lo);
    }

    double getHighComponent() {
        return hi;
    }

    double getLowComponent() {
        return lo;
    }

    public boolean gt(DoubleDouble y) {
        return (hi > y.hi) || (hi == y.hi && lo > y.lo);
    }

    public int intValue() {
        return (int) hi;
    }

    public boolean isInfinite() {
        return Double.isInfinite(hi);
    }

    public boolean isNaN() {
        return isNaN;
    }

    public boolean isNegative() {
        return hi < 0.0 || (hi == 0.0 && lo < 0.0);
    }

    public boolean isPositive() {
        return hi > 0.0 || (hi == 0.0 && lo > 0.0);
    }

    public boolean isZero() {
        return hi == 0.0 && lo == 0.0;
    }

    public boolean isInteger() {
        return ((hi == Math.floor(hi)) && !Double.isInfinite(hi)) && ((lo == Math.floor(
                lo)) && !Double.isInfinite(lo));
    }
    /*------------------------------------------------------------
     *   Conversion Functions
     *------------------------------------------------------------
     */

    public boolean le(DoubleDouble y) {
        return (hi < y.hi) || (hi == y.hi && lo <= y.lo);
    }

    public DoubleDouble log() {
        // Return the natural log of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }
        if (isZero()) {
            return NEGATIVE_INFINITY;
        }

        if (isNegative()) {
            return NaN;
        }

        DoubleDouble number = this;
        int intPart = 0;
        while (number.gt(E)) {
            number = number.divide(E);
            intPart++;
        }
        while (number.lt(E.reciprocal())) {
            number = number.multiply(E);
            intPart--;
        }

        final DoubleDouble num = number.subtract(ONE);
        final DoubleDouble denom = number.add(ONE);
        final DoubleDouble ratio = num.divide(denom);
        final DoubleDouble ratioSquare = ratio.multiply(ratio);
        DoubleDouble s = TWO.multiply(ratio);
        DoubleDouble sOld;
        DoubleDouble t = fromDoubleDouble(s);
        DoubleDouble w;
        double n = 1.0;

        do {
            n += 2.0;
            t = t.multiply(ratioSquare);
            w = t.divide(fromOneDouble(n));
            sOld = s;
            s = s.add(w);
        } while (s.ne(sOld));
        return s.add(fromOneDouble(intPart));
    }

    public DoubleDouble log10() {
        return this.log().divide(LOG_TEN);
    }

    /*------------------------------------------------------------
     *   Predicates
     *------------------------------------------------------------
     */

    public boolean lt(DoubleDouble y) {
        return (hi < y.hi) || (hi == y.hi && lo < y.lo);
    }

    public DoubleDouble max(DoubleDouble x) {
        if (this.ge(x)) {
            return this;
        } else {
            return x;
        }
    }

    public DoubleDouble min(DoubleDouble x) {
        if (this.le(x)) {
            return this;
        } else {
            return x;
        }
    }

    public boolean ne(DoubleDouble y) {
        return hi != y.hi || lo != y.lo;
    }

    public DoubleDouble negate() {
        if (isNaN) {
            return this;
        }
        return fromTwoDouble(-hi, -lo);
    }

    public DoubleDouble pow(DoubleDouble x) {
        DoubleDoubleCache<DoubleDouble> map = POW_DOUBLE_DOUBLE_CACHE.get(hi, lo,
                DoubleDoubleCache::new);
        return map.get(x.hi, x.lo, () -> innerPow(x));
    }

    private DoubleDouble innerPow(DoubleDouble x) {
        boolean invert = false;
        if (x.isNaN || x.isInfinite() || isInfinite() || isNaN) {
            return NaN;
        }
        if (x.isZero()) {
            return ONE;
        } else {
            if (isZero()) {
                return ZERO;
            } else if (isNegative()) {
                if (x.isInteger())
                    return pow(x.intValue());
                else
                    return NaN;
            } else {
                final DoubleDouble loga = this.log();
                DoubleDouble base = x.multiply(loga);
                if (base.lt(ZERO)) {
                    // Much greater precision if all numbers in the series have the same
                    // sign.
                    base = base.negate();
                    invert = true;
                }
                DoubleDouble s = ONE.add(base);
                DoubleDouble sOld;
                DoubleDouble t = fromDoubleDouble(base);
                double n = 1.0;

                do {
                    n += 1.0;
                    t = t.divide(fromOneDouble(n));
                    t = t.multiply(base);
                    sOld = s;
                    s = s.add(t);
                } while (s.ne(sOld));
                if (invert) {
                    s = s.reciprocal();
                }
                return s;
            }
        }
    }

    public DoubleDouble pow(int exp) {
        CacheMap<Integer, DoubleDouble> cacheMap = POW_INTEGER_CACHE.get(hi, lo,
                () -> new CacheMap<>(POW_INT_CACHE_SIZE_LIMIT));
        return cacheMap.get(exp, () -> innerPow(exp));
    }

    private DoubleDouble innerPow(int exp) {
        if (exp == 0.0) {
            return ONE;
        }

        DoubleDouble r = fromDoubleDouble(this);
        DoubleDouble s = ONE;
        int n = Math.abs(exp);

        if (n > 1) {
            /* Use binary exponentiation */
            while (n > 0) {
                if (n % 2 == 1) {
                    s = s.multiply(r);
                }
                n /= 2;
                if (n > 0) {
                    r = r.sqr();
                }
            }
        } else {
            s = r;
        }

        /* Compute the reciprocal if n is negative. */
        if (exp < 0) {
            return s.reciprocal();
        }
        return s;
    }

    public DoubleDouble reciprocal() {
        double hc, tc, hy, ty, bigC, c, bigU, u;
        bigC = 1.0 / hi;
        c = SPLIT * bigC;
        hc = c - bigC;
        u = SPLIT * hi;
        hc = c - hc;
        tc = bigC - hc;
        hy = u - hi;
        bigU = bigC * hi;
        hy = u - hy;
        ty = hi - hy;
        u = (((hc * hy - bigU) + hc * ty) + tc * hy) + tc * ty;
        c = (((1.0 - bigU) - u) - bigC * lo) / hi;

        final double zhi = bigC + c;
        final double zlo = (bigC - zhi) + c;
        return fromTwoDouble(zhi, zlo);
    }

    public DoubleDouble rint() {
        if (isNaN) {
            return this;
        }
        // may not be 100% correct
        final DoubleDouble plus5 = this.add(HALF);
        return plus5.floor();
    }

    public DoubleDouble add(DoubleDouble y) {
        double bigH, h, bigT, t, bigS, s, e, f;
        bigS = hi + y.hi;
        bigT = lo + y.lo;
        e = bigS - hi;
        f = bigT - lo;
        s = bigS - e;
        t = bigT - f;
        s = (y.hi - e) + (hi - s);
        t = (y.lo - f) + (lo - t);
        e = s + bigT;
        bigH = bigS + e;
        h = e + (bigS - bigH);
        e = t + h;

        double zhi = bigH + e;
        double zlo = e + (bigH - zhi);
        return fromTwoDouble(zhi, zlo);
    }

    /*------------------------------------------------------------
     *   Output
     *------------------------------------------------------------
     */

    public DoubleDouble multiply(DoubleDouble y) {
        double hx, tx, hy, ty, bigC, c;
        bigC = SPLIT * hi;
        hx = bigC - hi;
        c = SPLIT * y.hi;
        hx = bigC - hx;
        tx = hi - hx;
        hy = c - y.hi;
        bigC = hi * y.hi;
        hy = c - hy;
        ty = y.hi - hy;
        c = ((((hx * hy - bigC) + hx * ty) + tx * hy) + tx * ty) + (hi * y.lo + lo * y.hi);
        final double zhi = bigC + c;
        hx = bigC - zhi;
        final double zlo = c + hx;
        return fromTwoDouble(zhi, zlo);
    }

    public DoubleDouble si() {
        final DoubleDouble x = fromDoubleDouble(this);
        final DoubleDouble ci = ZERO;
        final DoubleDouble si = ZERO;
        cisia(x, ci, si);
        return si;
    }

    public int signum() {
        if (isPositive()) {
            return 1;
        }
        if (isNegative()) {
            return -1;
        }
        return 0;
    }

    public DoubleDouble sin() {
        boolean negate = false;
        // Return the sine of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }
        DoubleDouble twoPIFullTimes;
        DoubleDouble twoPIremainder;
        if ((this.abs()).gt(TWO_PI)) {
            twoPIFullTimes = (this.divide(TWO_PI)).trunc();
            twoPIremainder = this.subtract(TWO_PI.multiply(twoPIFullTimes));
        } else {
            twoPIremainder = this;
        }
        if (twoPIremainder.gt(PI)) {
            twoPIremainder = twoPIremainder.subtract(PI);
            negate = true;
        } else if (twoPIremainder.lt(PI.negate())) {
            twoPIremainder = twoPIremainder.add(PI);
            negate = true;
        }
        final DoubleDouble msquare = (twoPIremainder.multiply(twoPIremainder)).negate();
        DoubleDouble s = fromDoubleDouble(twoPIremainder);
        DoubleDouble sOld;
        DoubleDouble t = fromDoubleDouble(twoPIremainder);
        double n = 1.0;
        do {
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            t = t.multiply(msquare);
            sOld = s;
            s = s.add(t);
        } while (s.ne(sOld));
        if (negate) {
            s = s.negate();
        }
        return s;
    }

    public DoubleDouble sinh() {
        // Return the sinh of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }
        final DoubleDouble square = this.multiply(this);
        DoubleDouble s = fromDoubleDouble(this);
        DoubleDouble sOld;
        DoubleDouble t = fromDoubleDouble(this);
        double n = 1.0;
        do {
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            n += 1.0;
            t = t.divide(fromOneDouble(n));
            t = t.multiply(square);
            sOld = s;
            s = s.add(t);
        } while (s.ne(sOld));
        return s;
    }

    public DoubleDouble sqr() {
        return this.multiply(this);
    }

    public DoubleDouble sqrt() {
        /*
         * Strategy: Use Karp's trick: if x is an approximation to sqrt(a), then
         *
         * sqrt(a) = a*x + [a - (a*x)^2] * x / 2 (approx)
         *
         * The approximation is accurate to twice the accuracy of x. Also, the
         * multiplication (a*x) and [-]*x can be done with only half the
         * precision.
         */

        if (isZero()) {
            return ZERO;
        }

        if (isNegative()) {
            return NaN;
        }

        final double x = 1.0 / Math.sqrt(hi);
        final double ax = hi * x;

        final DoubleDouble axdd = fromOneDouble(ax);
        final DoubleDouble diffSq = this.subtract(axdd.sqr());
        final double d2 = diffSq.hi * (x * 0.5);

        return axdd.add(fromOneDouble(d2));
    }

    public DoubleDouble subtract(DoubleDouble y) {
        if (isNaN) {
            return this;
        }
        return add(y.negate());
    }

    public DoubleDouble tan() {
        // Return the tangent of a DoubleDouble number
        if (isNaN) {
            return NaN;
        }
        DoubleDouble piFullTimes;
        DoubleDouble piRemainder;
        if ((this.abs()).gt(PI)) {
            piFullTimes = (this.divide(PI)).trunc();
            piRemainder = this.subtract(PI.multiply(piFullTimes));
        } else {
            piRemainder = this;
        }
        if (piRemainder.gt(PI_2)) {
            piRemainder = piRemainder.subtract(PI);
        } else if (piRemainder.lt(PI_2.negate())) {
            piRemainder = piRemainder.add(PI);
        }
        if (piRemainder.equals(PI_2)) {
            return POSITIVE_INFINITY;
        } else if (piRemainder.equals(PI_2.negate())) {
            return NEGATIVE_INFINITY;
        }
        int twon;
        DoubleDouble twotwon;
        DoubleDouble twotwonm1;
        final DoubleDouble square = piRemainder.multiply(piRemainder);
        DoubleDouble s = fromDoubleDouble(piRemainder);
        DoubleDouble sOld;
        DoubleDouble t = fromDoubleDouble(piRemainder);
        int n = 1;
        do {
            n++;
            twon = 2 * n;
            t = t.divide(factorial(twon));
            twotwon = (TWO).pow(twon);
            twotwonm1 = twotwon.subtract(ONE);
            t = t.multiply(twotwon);
            t = t.multiply(twotwonm1);
            t = t.multiply(BernoulliB(n));
            t = t.multiply(square);
            sOld = s;
            s = s.add(t);
        } while (s.ne(sOld));
        return s;
    }

    @Override
    public String toString() {
        return Double.toString(doubleValue());
    }

    public DoubleDouble trunc() {
        if (isNaN) {
            return NaN;
        }
        if (isPositive()) {
            return floor();
        } else {
            return ceil();
        }
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(hi, lo);
    }

    @Override
    public boolean equals(Object object) {
        if (object instanceof DoubleDouble) {
            DoubleDouble that = (DoubleDouble) object;
            return DoubleMath.fuzzyEquals(hi, that.hi, 1E-12) && DoubleMath.fuzzyEquals(lo, that.lo,
                    1E-12);
        }
        return false;
    }

    public DoubleDouble exp() {
        return EXP_CACHE.get(hi, lo, this::innerExp);
    }
}
