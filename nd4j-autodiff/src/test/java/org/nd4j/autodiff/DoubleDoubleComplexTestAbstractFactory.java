package org.nd4j.autodiff;

import org.junit.Test;

public class DoubleDoubleComplexTestAbstractFactory
        extends AbstractFactoriesTest<DoubleDoubleComplex> {

    private static final double EQUAL_DELTA = 1e-12;

    public DoubleDoubleComplexTestAbstractFactory() {
        super(EQUAL_DELTA);
    }

    @Override
    protected AbstractFactory<DoubleDoubleComplex> getFactory() {
        return DoubleDoubleComplexFactory.instance();
    }

    @Test
    public void testComplexAcos() {
        assertComplexRelative(new DoubleDoubleComplex(2.88941324487344, -2.096596457288891),
                getFactory().acos(new DoubleDoubleComplex(-4, 1)));
        assertComplexRelative(new DoubleDoubleComplex(3.14159265358979, -2.063437068895560),
                getFactory().acos(new DoubleDoubleComplex(-4, 0)));
    }

    @Test
    public void testComplexAsin() {
        assertComplexRelative(new DoubleDoubleComplex(-1.57079632679489, 2.063437068895560),
                getFactory().asin(new DoubleDoubleComplex(-4, 0)));
        assertComplexRelative(new DoubleDoubleComplex(-1.31861691807854, 2.096596457288891),
                getFactory().asin(new DoubleDoubleComplex(-4, 1)));
    }

    @Test
    public void testComplexAtan() {
        assertComplexRelative(new DoubleDoubleComplex(-1.01722196789785, 0.402359478108525),
                getFactory().atan(new DoubleDoubleComplex(-1, 1)));
    }

    @Test
    public void testComplexCosh() {
        assertComplexRelative(new DoubleDoubleComplex(0.83373002513114, 0.98889770576286),
                getFactory().cosh(new DoubleDoubleComplex(1, 1)));
    }

    @Test
    public void testComplexSinh() {
        assertComplexRelative(new DoubleDoubleComplex(0.6349639147847361, 1.2984575814159772),
                getFactory().sinh(new DoubleDoubleComplex(1, 1)));
    }

    @Test
    public void testComplexTanh() {
        assertComplexRelative(new DoubleDoubleComplex(1.0839233273386945, 0.2717525853195117),
                getFactory().tanh(new DoubleDoubleComplex(1, 1)));
    }

    @Test
    public void testComplexAcosh() {
        assertComplexRelative(new DoubleDoubleComplex(2.2924316695611776, 3.1415926535897932),
                getFactory().acosh(new DoubleDoubleComplex(-5, 0)));
        assertComplexRelative(new DoubleDoubleComplex(1.0612750619050356, 0.9045568943023813),
                getFactory().acosh(new DoubleDoubleComplex(1, 1)));
    }

    @Test
    public void testComplexAsinh() {
        assertComplexRelative(new DoubleDoubleComplex(1.0612750619050356, 0.6662394324925152),
                getFactory().asinh(new DoubleDoubleComplex(1, 1)));
    }

    @Test
    public void testComplexAtanh() {
        assertComplexRelative(new DoubleDoubleComplex(0.5493061443340548, -1.5707963267948966),
                getFactory().atanh(new DoubleDoubleComplex(2, 0)));
        assertComplexRelative(new DoubleDoubleComplex(0.4023594781085250, 1.0172219678978513),
                getFactory().atanh(new DoubleDoubleComplex(1, 1)));
    }

    private void assertComplexRelative(DoubleDoubleComplex expected, DoubleDoubleComplex actual) {
        assertRelativeError(expected.getReal(), actual.getReal(), EQUAL_DELTA);
        assertRelativeError(expected.getImaginary(), actual.getImaginary(), EQUAL_DELTA);
    }
}
