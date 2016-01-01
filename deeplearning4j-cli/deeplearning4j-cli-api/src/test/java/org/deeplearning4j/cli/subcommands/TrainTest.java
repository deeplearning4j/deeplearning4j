package org.deeplearning4j.cli.subcommands;



import org.apache.commons.io.FileUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

public class TrainTest {

	@Test
    @Ignore
	public void testLoadInputFormat() throws Exception {
		
		System.out.println("[testLoadInputFormat] Start");
		

		//String testSVMLightInputFile = "src/test/resources/data/irisSvmLight.txt";
		File resource =  new ClassPathResource("confs/cli_train_unit_test_conf.txt").getFile();
        if(!resource.isDirectory())
            throw new IllegalStateException("Resolved to a directory");
		String conf_file = resource.getAbsolutePath();
        FileUtils.copyFile(new ClassPathResource("data/irisSvmLight.txt").getFile(),new File(System.getProperty("java.io.tmpdir"),"data/irisSvmLight.txt"));
        String[] args = { "-conf", conf_file ,"-input",conf_file};
		
		Train train = new Train(args);
		try {
			train.loadConfigFile();
		} catch (Exception e) {
			System.out.println( "could not load conf: " + e );
		}
		train.execute();
		
		System.out.println("[testLoadInputFormat] End");
		
		
	}

}
