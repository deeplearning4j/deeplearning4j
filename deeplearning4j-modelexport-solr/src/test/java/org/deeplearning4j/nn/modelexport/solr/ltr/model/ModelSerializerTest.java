package org.deeplearning4j.nn.modelexport.solr.ltr.model;

import java.io.File;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import org.junit.Test;
import static org.junit.Assert.assertEquals;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ModelSerializerTest {

  @Test
  public void test() throws Exception {

    final File tempFile = File.createTempFile("prefix", "suffix");
    tempFile.deleteOnExit();

    final MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .list(
            new OutputLayer.Builder().nIn(3).nOut(1).activation(Activation.IDENTITY).build()
            )
        .build();

    final MultiLayerNetwork originalModel = new MultiLayerNetwork(conf);
    originalModel.init();
    final INDArray params = Nd4j.create(new float[] { 2f, 4f, 8f, 16f }); // 3 weights + bias of 16
    originalModel.setParams(params);
    System.out.println("originalModel.params() = "+originalModel.params());

    for (final float val : new float[] { 1f, 0.5f, 0f }) {
      final float[] floats = new float[] { val, val, val };

      final INDArray input = Nd4j.create(floats);
      System.out.println("input = "+input);

      final INDArray originalOutput  = originalModel.output(input);
      System.out.println("originalOutput = "+originalOutput);
      final float    originalOutput0 = originalOutput.getFloat(0);
      System.out.println("originalOutput0 = "+originalOutput0);

      ModelSerializer.writeModel(originalModel, tempFile, false);

      final MultiLayerNetwork restoredModel = ModelSerializer.restoreMultiLayerNetwork(tempFile);

      final INDArray restoredOutput  = restoredModel.output(input);
      System.out.println("restoredOutput = "+restoredOutput);
      final float    restoredOutput0 = restoredOutput.getFloat(0);
      System.out.println("restoredOutput0 = "+restoredOutput0);

      assertEquals("input="+input+" originalOutput="+originalOutput+" restoredOutput="+restoredOutput, originalOutput0, restoredOutput0, 0f);
    }
  }

}
