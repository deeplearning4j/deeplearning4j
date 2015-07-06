package org.arbiter.cli.subcommands;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestEvaluate {


	/**
	 * Make sure we're loading the training process configuration file
	 * 
	 */
	@Test
	public void testLoadConf() {


		String conf_file = "src/test/resources/evaluate/conf/eval_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Evaluate cmd = new Evaluate( args );
		//cmd.execute();
		try {
			cmd.loadConfigFile();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		cmd.debugLoadedConfProperties();
		
		// dl4j.input.format
		String modelPath = cmd.configProps.getProperty(Evaluate.MODEL_PATH_KEY, "");
		assertEquals( "/tmp/foo.model", modelPath );
		
		
		
	}

	/**
	 * Make sure we're loading the network architecture json file
	 * 
	 */
	@Test
	public void testLoadModelArchitecture() {

/*
		String conf_file = "src/test/resources/train/architectures/dbn/conf/dbn_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Train cmd = new Train( args );
		//cmd.execute();
		try {
			cmd.loadConfigFile();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
		
		cmd.debugPrintConf();
		cmd.validateModelConfigFile();
		
		assertEquals( true, cmd.validModelConfigJSONFile );
	*/
	}
	
	/**
	 * Test loading the conf, network arch, and then train on a dataset
	 * 
	 */
	@Test
	public void testFull_Eval_ProcessDBNIris() {

		String conf_file = "src/test/resources/train/architectures/dbn/conf/dbn_test_conf.txt";
		
		String[] args = { "-conf", conf_file }; // ,"-input",conf_file};
		
		Evaluate cmd = new Evaluate( args );
		//cmd.execute();
	
	}
}
