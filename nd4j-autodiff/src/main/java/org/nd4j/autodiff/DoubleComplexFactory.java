package org.nd4j.autodiff;


public class DoubleComplexFactory implements AbstractIdentityFactory<DoubleComplex> {

    private static final DoubleComplexFactory m_INSTANCE = new DoubleComplexFactory();

    private DoubleComplexFactory() {
    }


    public static DoubleComplexFactory instance() {
        return m_INSTANCE;
    }


    public DoubleComplex create(double i_re, double i_im) {
        return new DoubleComplex(i_re, i_im);
    }

    private static final DoubleComplex m_ZERO = new DoubleComplex(0.0, 0.0);
    private static final DoubleComplex m_UNIT = new DoubleComplex(1.0, 0.0);

    public DoubleComplex zero() {
        return m_ZERO;
    }

    public DoubleComplex one() {
        return m_UNIT;
    }

}
