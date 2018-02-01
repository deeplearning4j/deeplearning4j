package org.deeplearning4j.regressiontest;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.*;
import org.junit.Test;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 08/05/2017.
 */
public class TestDistributionDeserializer extends BaseDL4JTest {

    @Test
    public void testDistributionDeserializer() throws Exception {
        //Test current format:
        Distribution[] distributions =
                        new Distribution[] {new NormalDistribution(3, 0.5), new UniformDistribution(-2, 1),
                                        new GaussianDistribution(2, 1.0), new BinomialDistribution(10, 0.3)};

        ObjectMapper om = NeuralNetConfiguration.mapper();

        for (Distribution d : distributions) {
            String json = om.writeValueAsString(d);
            Distribution fromJson = om.readValue(json, Distribution.class);

            assertEquals(d, fromJson);
        }
    }

    @Test
    public void testDistributionDeserializerLegacyFormat() throws Exception {
        ObjectMapper om = NeuralNetConfiguration.mapper();

        String normalJson = "{\n" + "          \"normal\" : {\n" + "            \"mean\" : 0.1,\n"
                        + "            \"std\" : 1.2\n" + "          }\n" + "        }";

        Distribution nd = om.readValue(normalJson, Distribution.class);
        assertTrue(nd instanceof NormalDistribution);
        NormalDistribution normDist = (NormalDistribution) nd;
        assertEquals(0.1, normDist.getMean(), 1e-6);
        assertEquals(1.2, normDist.getStd(), 1e-6);


        String uniformJson = "{\n" + "                \"uniform\" : {\n" + "                  \"lower\" : -1.1,\n"
                        + "                  \"upper\" : 2.2\n" + "                }\n" + "              }";

        Distribution ud = om.readValue(uniformJson, Distribution.class);
        assertTrue(ud instanceof UniformDistribution);
        UniformDistribution uniDist = (UniformDistribution) ud;
        assertEquals(-1.1, uniDist.getLower(), 1e-6);
        assertEquals(2.2, uniDist.getUpper(), 1e-6);


        String gaussianJson = "{\n" + "                \"gaussian\" : {\n" + "                  \"mean\" : 0.1,\n"
                        + "                  \"std\" : 1.2\n" + "                }\n" + "              }";

        Distribution gd = om.readValue(gaussianJson, Distribution.class);
        assertTrue(gd instanceof GaussianDistribution);
        GaussianDistribution gDist = (GaussianDistribution) gd;
        assertEquals(0.1, gDist.getMean(), 1e-6);
        assertEquals(1.2, gDist.getStd(), 1e-6);

        String bernoulliJson =
                        "{\n" + "                \"binomial\" : {\n" + "                  \"numberOfTrials\" : 10,\n"
                                        + "                  \"probabilityOfSuccess\" : 0.3\n" + "                }\n"
                                        + "              }";

        Distribution bd = om.readValue(bernoulliJson, Distribution.class);
        assertTrue(bd instanceof BinomialDistribution);
        BinomialDistribution binDist = (BinomialDistribution) bd;
        assertEquals(10, binDist.getNumberOfTrials());
        assertEquals(0.3, binDist.getProbabilityOfSuccess(), 1e-6);
    }

}
