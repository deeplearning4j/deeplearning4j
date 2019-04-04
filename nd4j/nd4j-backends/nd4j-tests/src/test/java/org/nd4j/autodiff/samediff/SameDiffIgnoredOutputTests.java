package org.nd4j.autodiff.samediff;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Alex on 04/04/2019.
 */
public class SameDiffIgnoredOutputTests {

    @Test
    public void testIgnoredOutput(){

        SameDiff sd = SameDiff.create();
        SDVariable ph1 = sd.placeHolder("ph", DataType.FLOAT, 3, 4);
        ph1.setArray(Nd4j.create(DataType.FLOAT, 3, 4));

        SDVariable add = ph1.add(1);

        SDVariable shape = add.shape();
        SDVariable out = add.sum();


        sd.createGradFunction();

    }

}
