package org.nd4j.autodiff;

public class DoubleComplex implements ComplexNumber<DoubleReal, DoubleComplex>, Cloneable {

    private double m_re;
    private double m_im;

    public DoubleComplex() {
        this(0.0, 0.0);
    }

    public DoubleComplex(double i_re, double i_im) {
        m_re = i_re;
        m_im = i_im;
    }

    public DoubleReal re() {
        return new DoubleReal(m_re);
    }

    public DoubleReal im() {
        return new DoubleReal(m_im);
    }

    public double re_double() {
        return m_re;
    }

    public double im_double() {
        return m_im;
    }

    public double modulus() {
        return Math.sqrt(absolute_square());
    }

    public double absolute_square() {
        return m_re * m_re + m_im * m_im;
    }

    public String toString() {
        return "(" + String.valueOf(m_re) + " + i " + String.valueOf(m_im) + ")";
    }

    public Object clone() {
        return new DoubleComplex(m_re, m_im);
    }

    public DoubleComplex conjugate() {
        return new DoubleComplex(m_re, -m_im);
    }

    public DoubleComplex inverse() {
        double r2 = absolute_square();
        return new DoubleComplex(m_re / r2, -m_re / r2);
    }

    public DoubleComplex negate() {
        return new DoubleComplex(-m_re, -m_im);
    }

    public DoubleComplex plus(DoubleComplex i_cd) {
        return new DoubleComplex(m_re + i_cd.m_re, m_im + i_cd.m_im);
    }

    public DoubleComplex minus(DoubleComplex i_cd) {
        return new DoubleComplex(m_re - i_cd.m_re, m_im - i_cd.m_im);
    }

    public DoubleComplex mul(DoubleComplex i_cd) {
        return new DoubleComplex((m_re * i_cd.m_re) - (m_im * i_cd.m_im),
                (m_re * i_cd.m_im) + (m_im * i_cd.m_re));
    }

    public DoubleComplex div(DoubleComplex i_cd) {
        return this.mul(i_cd.conjugate()).divide(i_cd.m_re * i_cd.m_re + i_cd.m_im * i_cd.m_im);
    }

    public DoubleComplex plus(double i_cd) {
        return new DoubleComplex(m_re + i_cd, m_im);
    }

    public DoubleComplex minus(double i_cd) {
        return new DoubleComplex(m_re - i_cd, m_im);
    }

    public DoubleComplex prod(double i_cd) {
        return new DoubleComplex(m_re * i_cd, m_im * i_cd);
    }

    public DoubleComplex divide(double i_cd) {
        return new DoubleComplex(m_re / i_cd, m_im / i_cd);
    }

    public DoubleComplex pow(int i_n) {
        double abs = this.absolute_square();
        return new DoubleComplex(abs * Math.cos(((double) i_n) * Math.acos(m_re / abs)),
                abs * Math.sin(((double) i_n) * Math.asin(m_im / abs)));
    }

    public DoubleComplex mul(long i_n) {
        return new DoubleComplex(m_re * i_n, m_im * i_n);
    }

    @Override
    public double getReal() {
        // TODO Auto-generated method stub
        return 0;
    }
}
