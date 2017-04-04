package org.nd4j.autodiff;

public class DoubleRealTestAbstractFactory extends AbstractFactoriesTest<DoubleReal> {

    public DoubleRealTestAbstractFactory() {
        super(1e-15);
    }

    @Override
    protected AbstractFactory<DoubleReal> getFactory() {
        return DoubleRealFactory.instance();
    }
}
