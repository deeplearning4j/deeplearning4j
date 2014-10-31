package org.nd4j.linalg.jblas.util;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/31/14.
 */
public class FeatureUtilTest {

    private static Logger log = LoggerFactory.getLogger(FeatureUtil.class);


    @Test
    public void testMinMax() {
        INDArray test = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray assertion = Nd4j.create(Nd4j.createBuffer(new double[]{0,1,0,1}),new int[]{2,2});
        FeatureUtil.scaleMinMax(0,1,test);
        assertEquals(assertion,test);
    }


}
