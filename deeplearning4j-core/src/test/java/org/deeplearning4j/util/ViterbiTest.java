package org.deeplearning4j.util;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 5/5/14.
 */
public class ViterbiTest {

    private static Logger log = LoggerFactory.getLogger(ViterbiTest.class);

    @Test
    public void viterbiTest() {
        Viterbi v = new Viterbi(new DoubleMatrix(new double[]{1,2}));
        DoubleMatrix label = new DoubleMatrix(new double[][]{
                {1},{2},{2},{1}
        });

        log.info(String.valueOf(v.decode(label,false)));
    }

}
