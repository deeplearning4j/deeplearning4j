package org.deeplearning4j.nn.conf.preprocessor;

import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.introspect.AnnotatedClass;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.custom.MyCustomPreprocessor;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 09/09/2016.
 */
public class CustomPreprocessorTest {

    @Test
    public void testCustomPreprocessor(){
        //First: Ensure that the CustomLayer class is registered
        ObjectMapper mapper = NeuralNetConfiguration.mapper();

        AnnotatedClass ac = AnnotatedClass.construct(InputPreProcessor.class, mapper.getSerializationConfig().getAnnotationIntrospector(), null);
        Collection<NamedType> types = mapper.getSubtypeResolver().collectAndResolveSubtypes(ac, mapper.getSerializationConfig(), mapper.getSerializationConfig().getAnnotationIntrospector());
        boolean found = false;
        for (NamedType nt : types) {
//            System.out.println(nt);
            if (nt.getType() == MyCustomPreprocessor.class){
                found = true;
                break;
            }
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
