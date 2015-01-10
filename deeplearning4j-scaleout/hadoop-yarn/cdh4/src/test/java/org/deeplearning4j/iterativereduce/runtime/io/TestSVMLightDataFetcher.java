package org.deeplearning4j.iterativereduce.runtime.io;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;

import org.nd4j.linalg.util.FeatureUtil;

public class TestSVMLightDataFetcher {


	private static String svmLight_test_filename = "src/test/resources/data/svmLightSample.txt";
	

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
		
	
	/**
	 * Currently we only support non-negative labels
	 * - this is due to the issue where we have not yet accounted for the scenario
	 *   where each split needs the same label conversion heuristic 
	 *   we're just using the class labels directly as indexes
	 * 
	 */
	@Test
	public void testSVMLightHDFSFetcher() {
		
		// setup splits ala HDFS style -------------------
		
	    JobConf job = new JobConf(defaultConf);
	    
	    Path workDir = new Path( svmLight_test_filename );
				
		
	    InputSplit[] splits = generateDebugSplits(workDir, job);
	    
	    System.out.println( "> splits: " + splits[0].toString() );

	    
	    TextRecordParser txt_reader = new TextRecordParser();

	    long len = Integer.parseInt(splits[0].toString().split(":")[2]
	        .split("\\+")[1]);

	    txt_reader.setFile(splits[0].toString().split(":")[1], 0, len);		
		
	    SVMLightDataFetcher fetcher = null;
	    
	    try {
			fetcher = new SVMLightDataFetcher( txt_reader, 9300, 2 );
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	    
	    
	    fetcher.fetch( 20 );
	    DataSet ds = fetcher.next();
	    
	    
	    INDArray features = ds.getFeatures();
	    INDArray labels = ds.getLabels();
	    
	    
	    //System.out.println( "feature columns: " + features.columns() );
	    
	    assertEquals( 9300, features.columns() );
	    assertEquals( 2, labels.columns() );
				

/*	    
		MnistHDFSDataSetIterator hdfs_fetcher = new MnistHDFSDataSetIterator( batchSize, totalNumExamples, txt_reader );
		DataSet hdfs_recordBatch = hdfs_fetcher.next();
		
		Matrix hdfs_input = hdfs_recordBatch.getFirst();
		Matrix hdfs_labels = hdfs_recordBatch.getSecond();		
		
		// setup splits ala HDFS style -------------------
		
		
		// now download the binary data if needed 
		
		MNIST_DatasetUtils util = new MNIST_DatasetUtils();
		util.convertFromBinaryFormatToMetronome( 5, vectors_filename );
		*/
		/*
		assertEquals( hdfs_input.numCols(), stock_input.numCols() );
		assertEquals( hdfs_input.numRows(), stock_input.numRows() );		
		*/
		
		
		
	}

}
