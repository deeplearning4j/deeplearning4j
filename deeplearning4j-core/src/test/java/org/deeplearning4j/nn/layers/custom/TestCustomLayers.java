/*
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.deeplearning4j.nn.layers.custom;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 26/08/2016.
 */
public class TestCustomLayers {

    @Test
    public void testJsonRegistration(){

        //First: Ensure that the CustomTestLayer class is registered
        ObjectMapper mapper = NeuralNetConfiguration.mapper();

        AnnotatedClass ac = AnnotatedClass.construct(Layer.class, mapper.getSerializationConfig().getAnnotationIntrospector(), null);
        Collection<NamedType> types = mapper.getSubtypeResolver().collectAndResolveSubtypes(ac, mapper.getSerializationConfig(), mapper.getSerializationConfig().getAnnotationIntrospector());
        Set<Class<?>> registeredSubtypes = new HashSet<>();
        boolean found = false;
        for (NamedType nt : types) {
            System.out.println(nt);
//            registeredSubtypes.add(nt.getType());
            if(nt.getType() == CustomTestLayer.class) found = true;
        }

        assertTrue("CustomTestLayer: not registered with NeuralNetConfiguration mapper",found);

        //Second: let's create a MultiLayerCofiguration with one, and check JSON and YAML config actually works...

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .learningRate(0.1)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(1, new CustomTestLayer(3.14159))
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(10).nOut(10).build())
                .pretrain(false).backprop(true).build();

        String json = conf.toJson();
        String yaml = conf.toYaml();

        System.out.println(json);

        MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf, confFromJson);

        MultiLayerConfiguration confFromYaml = MultiLayerConfiguration.fromYaml(yaml);
        assertEquals(conf, confFromYaml);

    }

}
