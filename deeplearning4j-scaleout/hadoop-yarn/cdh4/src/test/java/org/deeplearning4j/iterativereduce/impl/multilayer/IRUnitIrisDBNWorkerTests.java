package org.deeplearning4j.iterativereduce.impl.multilayer;

import static org.junit.Assert.*;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.deeplearning4j.iterativereduce.irunit.IRUnitDriver;
import org.deeplearning4j.iterativereduce.runtime.io.TextRecordParser;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;


public class IRUnitIrisDBNWorkerTests {

	

	  private static JobConf defaultConf = new JobConf();
	  private static FileSystem localFs = null; 
	  static {
	    try {
	      defaultConf.set("fs.defaultFS", "file:///");
	      localFs = FileSystem.getLocal(defaultConf);
	    } catch (IOException e) {
	      throw new RuntimeException("init failure", e);
	    }
	  }
	
	
	private InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

		long block_size = localFs.getDefaultBlockSize();

		System.out.println("default block size: " + (block_size / 1024 / 1024)
				+ "MB");

		// ---- set where we'll read the input files from -------------
		FileInputFormat.setInputPaths(job, input_path);

		// try splitting the file in a variety of sizes
		TextInputFormat format = new TextInputFormat();
		format.configure(job);

		int numSplits = 1;

		InputSplit[] splits = null;

		try {
			splits = format.getSplits(job, numSplits);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return splits;

	}		
	/*
	public DataSet setupHDFSDataset( String vectors_filename ) throws IOException {
		
		
		int batchSize = 20;
		int totalNumExamples = 20;

		// setup splits ala HDFS style -------------------
		
	    JobConf job = new JobConf(defaultConf);
	    
	    Path workDir = new Path( vectors_filename );
		
		
	    InputSplit[] splits = generateDebugSplits(workDir, job);
	    
	    System.out.println( "> splits: " + splits[0].toString() );

	    
	    TextRecordParser txt_reader = new TextRecordParser();

	    long len = Integer.parseInt(splits[0].toString().split(":")[2]
	        .split("\\+")[1]);

	    txt_reader.setFile(splits[0].toString().split(":")[1], 0, len);		
		
				

	    
		MnistHDFSDataSetIterator hdfs_fetcher = new MnistHDFSDataSetIterator( batchSize, totalNumExamples, txt_reader );
		DataSet hdfs_recordBatch = hdfs_fetcher.next();
		
		return hdfs_recordBatch;
	}	
	*/
	
	@Test
	public void testSingleWorkerConfigSetup() {
		
	}

	@Test
	public void testSingleWorker() {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/yarn/configurations/svmLightWorkerIRUnitTest.properties");
		polr_ir.Setup();
		polr_ir.simulateRun();
		
		
	}
	
	@Test
	public void testTwoWorkers() {

		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/yarn/configurations/svmLightIris_TwoWorkers_IRUnitTest.properties");
		polr_ir.Setup();
		polr_ir.simulateRun();

		
	}

	@Test
	public void testThreeWorkers() {
		
		IRUnitDriver polr_ir = new IRUnitDriver("src/test/resources/yarn/configurations/svmLightIris_ThreeWorkers_IRUnitTest.properties");
		polr_ir.Setup();
		polr_ir.simulateRun();
		
		
	}
	
	
	@Test
	public void testConfIssues() {
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .list(4).hiddenLayerSizes(new int[]{3,2,2}).build();
			String json = conf.toJson();		
		
		Configuration c = new Configuration();
		c.set( "test_json", json );
		
		String key = c.get("test_json");
	    assertEquals(json,key);		
	    
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson( key );
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf2);

		
		
	}
	
	@Test
	public void loadFromFileProps() {
		
		String props_file = "src/test/resources/yarn/configurations/svmLightWorkerIRUnitTest.properties";
		
	    Properties props = new Properties();

	    try {
	      FileInputStream fis = new FileInputStream( props_file );
	      props.load(fis);
	      fis.close();
	    } catch (FileNotFoundException ex) {
	      // throw ex; // TODO: be nice
	      System.out.println(ex);
	    } catch (IOException ex) {
	      // throw ex; // TODO: be nice
	      System.out.println(ex);
	    }		
	    
	    String json = props.getProperty("org.deeplearning4j.scaleout.multilayerconf");
	    
        MultiLayerConfiguration conf2 = MultiLayerConfiguration.fromJson( json );
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(conf2);
	    
	    
		
	}

}
