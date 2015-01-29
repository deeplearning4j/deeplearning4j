package org.deeplearning4j.nn.conf;

import static org.junit.Assert.*;

import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;

import java.io.*;
import java.util.Properties;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerNeuralNetConfigurationTest {

    @Test
    public void testJson() throws Exception {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().layerFactory(LayerFactories.getFactory(RBM.class))
                .list(4).hiddenLayerSizes(new int[]{3,2,2}).build();
        String json = conf.toJson();
        MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
        assertEquals(conf,from);

        Properties props = new Properties();
        props.put("json",json);
        String key = props.getProperty("json");
        assertEquals(json,key);
        File f = new File("props");
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        props.store(bos,"");
        bos.flush();
        bos.close();
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(f));
        Properties props2 = new Properties();
        props2.load(bis);
        bis.close();
        assertEquals(props2.getProperty("json"),props.getProperty("json"));

        MultiLayerConfiguration conf3 = MultiLayerConfiguration.fromJson(props2.getProperty("json"));
        assertEquals(conf,conf3);


    }


}
