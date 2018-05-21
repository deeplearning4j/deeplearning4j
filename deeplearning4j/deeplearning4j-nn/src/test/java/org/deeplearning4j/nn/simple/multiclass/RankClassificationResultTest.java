package org.deeplearning4j.nn.simple.multiclass;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 4/28/17.
 */
public class RankClassificationResultTest {
    @Test
    public void testOutcome() {
        RankClassificationResult result =
                        new RankClassificationResult(Transforms.sigmoid(Nd4j.linspace(1, 4, 4)).reshape(2, 2));
        assumeNotNull(result.getLabels());
        assertEquals("1", result.maxOutcomeForRow(0));
        assertEquals("1", result.maxOutcomeForRow(1));
        List<String> maxOutcomes = result.maxOutcomes();
        assertEquals(2, result.maxOutcomes().size());
        for (int i = 0; i < 2; i++) {
            assertEquals("1", maxOutcomes.get(i));
        }
    }


}
