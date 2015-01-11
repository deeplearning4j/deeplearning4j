package org.deeplearning4j.iterativereduce.impl.multilayer;

import static org.junit.Assert.*;

import org.deeplearning4j.iterativereduce.irunit.IRUnitDriver;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.junit.Test;


public class IRUnitSVMLightWorkerTest {


	public void createSynthJSONConf() {
		
		 MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
         .list(4).hiddenLayerSizes(new int[]{3,2,2}).build();
		 String json = conf.toJson();
		 
		 System.out.println( json );
		 
		 
		 MultiLayerConfiguration from = MultiLayerConfiguration.fromJson(json);
		 assertEquals(conf,from);		
		
	}
	
	@Test
	public void testLearnIrisFunctionViaIRNN_MLP() throws Exception {
		
		//this.createSynthJSONConf();
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/yarn/configurations/svmLightWorkerIRUnitTest.properties");
		
		
		polr_ir.Setup();

		polr_ir.SimulateRun();


		
		
	}

}
