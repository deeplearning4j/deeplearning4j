package org.deeplearning4j.regressiontest;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.regressiontest.customlayer100a.CustomLayer;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.RmsProp;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class RegressionTest100a extends BaseDL4JTest {

    @Test
    public void testCustomLayer() throws Exception {

        File f = new ClassPathResource("CustomLayerExample_100a.bin").getTempFileFromArchive();

        try {
            MultiLayerNetwork.load(f, true);
            fail("Expected exception");
        } catch (Exception e){
            String msg = e.getMessage();
            assertTrue(msg, msg.contains("NeuralNetConfiguration.registerLegacyCustomClassesForJSON"));
        }

        NeuralNetConfiguration.registerLegacyCustomClassesForJSON(CustomLayer.class);

        MultiLayerNetwork net = MultiLayerNetwork.load(f, true);

        DenseLayer l0 = (DenseLayer) net.getLayer(0).conf().getLayer();
        assertEquals(new ActivationTanH(), l0.getActivationFn());
        assertEquals(0.03, l0.getL2(), 1e-6);
        assertEquals(new RmsProp(0.95), l0.getIUpdater());

        CustomLayer l1 = (CustomLayer) net.getLayer(1).conf().getLayer();
        assertEquals(new ActivationTanH(), l1.getActivationFn());
        assertEquals(new ActivationSigmoid(), l1.getSecondActivationFunction());
        assertEquals(new RmsProp(0.95), l1.getIUpdater());


        INDArray outExp;
        File f2 = new ClassPathResource("CustomLayerExample_Output_100a.bin").getTempFileFromArchive();
        try(DataInputStream dis = new DataInputStream(new FileInputStream(f2))){
            outExp = Nd4j.read(dis);
        }

        INDArray in;
        File f3 = new ClassPathResource("CustomLayerExample_Input_100a.bin").getTempFileFromArchive();
        try(DataInputStream dis = new DataInputStream(new FileInputStream(f3))){
            in = Nd4j.read(dis);
        }

        INDArray outAct = net.output(in);

        assertEquals(outExp, outAct);
    }

}
