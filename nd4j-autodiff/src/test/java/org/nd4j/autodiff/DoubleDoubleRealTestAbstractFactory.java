package org.nd4j.autodiff;

public class DoubleDoubleRealTestAbstractFactory extends AbstractFactoriesTest<DoubleDoubleReal> {

    public DoubleDoubleRealTestAbstractFactory() {
        super(1e-15);
    }

    @Override
    protected AbstractFactory<DoubleDoubleReal> getFactory() {
        return DoubleDoubleRealFactory.instance();
    }
}
