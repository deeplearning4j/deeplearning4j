package org.nd4j.autodiff.bigdecimal;

import java.math.BigDecimal;
import java.math.MathContext;


public class BigComplex {

    BigDecimal re;


    BigDecimal im;


    final static BigComplex ZERO = new BigComplex(BigDecimal.ZERO, BigDecimal.ZERO);


    public BigComplex() {
        re = BigDecimal.ZERO;
        im = BigDecimal.ZERO;
    }


    public BigComplex(BigDecimal x, BigDecimal y) {
        re = x;
        im = y;
    }


    public BigComplex(BigDecimal x) {
        re = x;
        im = BigDecimal.ZERO;
    }


    public BigComplex(double x, double y) {
        re = new BigDecimal(x);
        im = new BigDecimal(y);
    }


    BigComplex multiply(final BigComplex oth, MathContext mc) {
        final BigDecimal a = re.add(im).multiply(oth.re);
        final BigDecimal b = oth.re.add(oth.im).multiply(im);
        final BigDecimal c = oth.im.subtract(oth.re).multiply(re);
        final BigDecimal x = a.subtract(b, mc);
        final BigDecimal y = a.add(c, mc);
        return new BigComplex(x, y);
    }


    BigComplex add(final BigDecimal oth) {
        final BigDecimal x = re.add(oth);
        return new BigComplex(x, im);
    }


    BigComplex subtract(final BigComplex oth) {
        final BigDecimal x = re.subtract(oth.re);
        final BigDecimal y = im.subtract(oth.im);
        return new BigComplex(x, y);
    }


    BigComplex conj() {
        return new BigComplex(re, im.negate());
    }


    BigDecimal norm() {
        return re.multiply(re).add(im.multiply(im));
    }


    BigDecimal abs(MathContext mc) {
        return BigDecimalMath.sqrt(norm(), mc);
    }


    BigComplex sqrt(MathContext mc) {
        final BigDecimal half = new BigDecimal("2");
        /*
         * compute l=sqrt(re^2+im^2), then u=sqrt((l+re)/2) and v= +-
         * sqrt((l-re)/2 as the new real and imaginary parts.
         */
        final BigDecimal l = abs(mc);
        if (l.compareTo(BigDecimal.ZERO) == 0)
            return new BigComplex(BigDecimalMath.scalePrec(BigDecimal.ZERO, mc),
                    BigDecimalMath.scalePrec(BigDecimal.ZERO, mc));
        final BigDecimal u = BigDecimalMath.sqrt(l.add(re).divide(half, mc), mc);
        final BigDecimal v = BigDecimalMath.sqrt(l.subtract(re).divide(half, mc), mc);
        if (im.compareTo(BigDecimal.ZERO) >= 0)
            return new BigComplex(u, v);
        else
            return new BigComplex(u, v.negate());
    }


    BigComplex inverse(MathContext mc) {
        final BigDecimal hyp = norm();
        /* 1/(x+iy)= (x-iy)/(x^2+y^2 */
        return new BigComplex(re.divide(hyp, mc), im.divide(hyp, mc).negate());
    }


    BigComplex divide(BigComplex oth, MathContext mc) {
        /* lazy implementation: (x+iy)/(a+ib)= (x+iy)* 1/(a+ib) */
        return multiply(oth.inverse(mc), mc);
    }


    public String toString() {
        return "(" + re.toString() + "," + im.toString() + ")";
    }


    public String toString(MathContext mc) {
        return "(" + re.round(mc).toString() + "," + im.round(mc).toString() + ")";
    }

} /* BigComplex */
