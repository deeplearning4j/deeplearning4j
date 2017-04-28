package org.deeplearning4j.nn.simple.multiclass;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 4/28/17.
 */
public class MultiClassResultTest {
    @Test
    public void testOutcome() {
        MultiClassResult result = new MultiClassResult(Transforms.sigmoid(Nd4j.linspace(1,4,4)).reshape(2,2));
        assumeNotNull(result.getLabels());
        assertEquals("1",result.maxOutcomeForRow(0));
        assertEquals("1",result.maxOutcomeForRow(1));

        System.out.println(Arrays.toString(result.getRankedIndices()));
    }


}
