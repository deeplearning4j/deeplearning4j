package org.deeplearning4j.cli.subcommands;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;

public class TrainTest {

	@Test
	public void testLoadInputFormat() throws Exception {
		
		System.out.println("[testLoadInputFormat] Start");
		
		// org.canova.api.formats.input.impl.SVMLightInputFormat
		
		//String testSVMLightInputFile = "src/test/resources/data/irisSvmLight.txt";
		
		String conf_file = "src/test/resources/confs/cli_train_unit_test_conf.txt";
		
		String[] args = { "-conf", conf_file ,"-input",conf_file};
		
		Train train = new Train( args );
		try {
			train.loadConfigFile();
		} catch (Exception e) {
			System.out.println( "could not load conf: " + e );
		}
		train.debugLoadedConfProperties();
		train.exec();
		
		System.out.println("[testLoadInputFormat] End");
		
		
	}

}
