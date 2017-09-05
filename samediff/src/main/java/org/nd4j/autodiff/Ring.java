package org.nd4j.autodiff;


public interface Ring extends CommutativeGroup {

    ArrayField muli(ArrayField i_v);

    ArrayField muli(double v);

    ArrayField powi(int i_n);

    ArrayField mul(ArrayField i_v);

    ArrayField mul(double v);

    ArrayField pow(int i_n);

}
