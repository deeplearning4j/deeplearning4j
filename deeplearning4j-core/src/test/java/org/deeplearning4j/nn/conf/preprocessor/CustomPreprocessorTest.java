package org.deeplearning4j.nn.conf.preprocessor;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.custom.MyCustomPreprocessor;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 09/09/2016.
 */
public class CustomPreprocessorTest {

    @Test
    public void testCustomPreprocessor(){
        //First: Ensure that the CustomLayer class is registered
        boolean found = false;
        for(Class<?> c : NeuralNetConfiguration.getRegisteredSubtypes()){
            System.out.println(c);
            if (c == MyCustomPreprocessor.class) found = true;
        }

        assertTrue("MyCustomPreprocessor: not registered with NeuralNetConfiguration mapper", found);

        //Second: let's create a MultiLayerCofiguration with one, and check JSON and YAML config actually works...
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(10).nOut(10).build())
                .inputPreProcessor(0, new MyCustomPreprocessor())
                .pretrain(false).backprop(true).build();

        String json = conf.toJson();
        String yaml = conf.toYaml();

        System.out.println(json);

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, confFromJson);

        MultiLayerConfiguration confFromYaml = MultiLayerConfiguration.fromYaml(yaml);
        assertEquals(conf, confFromYaml);

        assertTrue(confFromJson.getInputPreProcess(0) instanceof MyCustomPreprocessor);

    }

}
