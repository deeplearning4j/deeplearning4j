package org.deeplearning4j.largevis;

import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertNotNull;

public class LargeVisTest {

    @Test
    public void testLargeVisRun() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        DataSet iris = new IrisDataSetIterator(150,150).next();
        LargeVis largeVis = LargeVis.builder()
                .vec(iris.getFeatureMatrix())
                .normalize(true)
                .seed(42).build();
        largeVis.fit();
        assertNotNull(largeVis.getResult());


    }

}
