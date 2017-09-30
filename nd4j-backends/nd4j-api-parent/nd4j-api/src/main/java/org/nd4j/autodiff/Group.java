package org.nd4j.autodiff;


public interface Group {


    ArrayField negate();


    ArrayField add(ArrayField i_v);

    ArrayField add(double i_v);

    ArrayField sub(ArrayField i_v);


    ArrayField rsub(ArrayField i_v);

    ArrayField mul(double i_n);

    ArrayField mul(ArrayField i_n);

    ArrayField sub(double i_v);

    ArrayField negatei();


    ArrayField addi(ArrayField i_v);

    ArrayField addi(double i_v);

    ArrayField subi(ArrayField i_v);


    ArrayField rsubi(ArrayField i_v);

    ArrayField muli(double i_n);

    ArrayField subi(double i_v);
}
