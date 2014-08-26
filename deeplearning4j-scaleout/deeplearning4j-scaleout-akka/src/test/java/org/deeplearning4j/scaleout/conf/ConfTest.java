package org.deeplearning4j.scaleout.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.rbm.RBM;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;

/**
 * @author Adam Gibson
 */
public class ConfTest {
    private static Logger log = LoggerFactory.getLogger(ConfTest.class);

    @Test
    public void testInit() {
        Conf c = new Conf();
        c.setSparsity(1e-2f);
        c.setDropOut(0.5f);
        c.setUseRegularization(true);
        c.setK(1);
        c.setHiddenUnitByLayer(Collections.singletonMap(0, RBM.HiddenUnit.RECTIFIED));
        c.setVisibleUnitByLayer(Collections.singletonMap(0, RBM.VisibleUnit.GAUSSIAN));
        c.setMultiLayerClazz(DBN.class);
        c.setLayerSizes(new int[]{300,300,300});
        c.setnOut(2);
        c.setnIn(2);
        c.setUseAdaGrad(true);


        BaseMultiLayerNetwork build = c.init();

        DBN d = new DBN.Builder().withHiddenUnitsByLayer(Collections.singletonMap(0, RBM.HiddenUnit.RECTIFIED))
               .withVisibleUnitsByLayer(Collections.singletonMap(0, RBM.VisibleUnit.GAUSSIAN))
                .withSparsity(1e-2f).useRegularization(true)
                .withDropOut(0.5f).hiddenLayerSizes(new int[]{300,300,300})
                .numberOfInputs(2).numberOfOutPuts(2).useAdaGrad(true).build();

        assertEquals(build,d);

    }

}
