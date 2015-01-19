package org.deeplearning4j.iterativereduce.impl.multilayer;

import static org.junit.Assert.*;

import org.deeplearning4j.iterativereduce.irunit.IRUnitDriver;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.junit.Test;


public class IRUnitSVMLightWorkerTest {


	@Test
	public void createSynthJSONConf() {
		
		 MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().nIn(4).nOut(3)
				 .layerFactory(LayerFactories.getFactory(RBM.class))
         .list(3).hiddenLayerSizes(new int[]{2,2}).build();
		 String json = conf.toJson();
		 
		 System.out.println( json );
		 
		 
		 MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
		 assertEquals(conf,from);		
		
	}
	
	@Test
	public void testLearnIrisFunctionViaIRNN_MLP() throws Exception {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/yarn/configurations/svmLightWorkerIRUnitTest.properties");
		
		polr_ir.setup();
		polr_ir.simulateRun();


		
		
	}

}
