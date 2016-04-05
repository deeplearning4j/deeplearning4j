package org.deeplearning4j.ui;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.feedforward.autoencoder.AutoEncoder;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.ui.weights.ModelAndGradient;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.serde.jackson.VectorDeSerializer;
import org.nd4j.serde.jackson.VectorSerializer;

import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class TestSerialization {
    @Test
    public void testModelSerde() throws Exception {
        ObjectMapper mapper = getMapper();
        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().momentum(0.9f)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1000)
                .learningRate(1e-1f)
                .layer(new org.deeplearning4j.nn.conf.layers.AutoEncoder.Builder()
                        .nIn(4).nOut(3)
                        .corruptionLevel(0.6)
                        .sparsity(0.5)
                        .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).build())
                .build();


        DataSet d2 = new IrisDataSetIterator(150,150).next();

        INDArray input = d2.getFeatureMatrix();
        AutoEncoder da = LayerFactories.getFactory(conf.getLayer()).create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1), new HistogramIterationListener(1)),0);
        da.setInput(input);
        ModelAndGradient g = new ModelAndGradient(da);
        String json = mapper.writeValueAsString(g);
        ModelAndGradient read = mapper.readValue(json,ModelAndGradient.class);
        assertEquals(g,read);
    }


    public ObjectMapper getMapper() {
        ObjectMapper mapper = new ObjectMapper();
        SimpleModule nd4j = new SimpleModule("nd4j");
        nd4j.addDeserializer(INDArray.class, new VectorDeSerializer());
        nd4j.addSerializer(INDArray.class, new VectorSerializer());
        mapper.registerModule(nd4j);
        return mapper;
    }


}
