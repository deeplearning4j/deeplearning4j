package org.nd4j.linalg.lossfunctions;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.api.ops.random.impl.BernoulliDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;

import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 09/09/2016.
 */
public class LossFunctionTest extends BaseNd4jTest {

    public LossFunctionTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testClippingXENT() throws Exception {

        ILossFunction l1 = new LossBinaryXENT(0);
        ILossFunction l2 = new LossBinaryXENT();

        INDArray labels = Nd4j.getExecutioner().exec(new BernoulliDistribution(Nd4j.create(3,5), 0.5));
        INDArray preOut = Nd4j.valueArrayOf(3, 5, -1000.0);

        IActivation a = new ActivationSigmoid();

        double score1 = l1.computeScore(labels, preOut.dup(), a, null, false);
        assertTrue(Double.isNaN(score1));

        double score2 = l2.computeScore(labels, preOut.dup(), a, null, false);
        assertFalse(Double.isNaN(score2));

        INDArray grad1 = l1.computeGradient(labels, preOut.dup(), a, null);
        INDArray grad2 = l2.computeGradient(labels, preOut.dup(), a, null);

        MatchCondition c1 = new MatchCondition(grad1, Conditions.isNan());
        MatchCondition c2 = new MatchCondition(grad2, Conditions.isNan());
        int match1 = Nd4j.getExecutioner().exec(c1, Integer.MAX_VALUE).getInt(0);
        int match2 = Nd4j.getExecutioner().exec(c2, Integer.MAX_VALUE).getInt(0);

        assertTrue(match1 > 0);
        assertEquals(0, match2);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
