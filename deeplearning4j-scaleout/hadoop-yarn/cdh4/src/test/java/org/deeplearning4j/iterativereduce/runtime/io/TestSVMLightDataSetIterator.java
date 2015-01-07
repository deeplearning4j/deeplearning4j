package org.deeplearning4j.iterativereduce.runtime.io;

import static org.junit.Assert.*;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;

public class TestSVMLightDataSetIterator {


	private static String svmLight_test_filename = "src/test/resources/svmLightSample.txt";
	

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
	
	@Test
	public void testBasicMechanics() throws IOException {
		
		// setup splits ala HDFS style -------------------
		
	    JobConf job = new JobConf(defaultConf);
	    
	    Path workDir = new Path( svmLight_test_filename );
				
		
	    InputSplit[] splits = generateDebugSplits(workDir, job);
	    
	    System.out.println( "> splits: " + splits[0].toString() );

	    
	    TextRecordParser txt_reader = new TextRecordParser();

	    long len = Integer.parseInt(splits[0].toString().split(":")[2]
	        .split("\\+")[1]);

	    txt_reader.setFile(splits[0].toString().split(":")[1], 0, len);		
				
		
		BaseDatasetIterator iterator = new SVMLightHDFSDataSetIterator( 20, 1, txt_reader, 9300, 2 );
		
		assertEquals( true, iterator.hasNext() );
		DataSet ds = iterator.next();
		//System.out.println( "rows" + ds.getLabels().rows() );
		//System.out.println( "cols" + ds.getLabels().columns() );
		assertEquals( 1.0, ds.getLabels().getRow(0).getDouble(0), 0.0 );
		assertEquals( 0.0, ds.getLabels().getRow(0).getDouble(1), 0.0 );
		
		assertEquals( 0.0, ds.getLabels().getRow(1).getDouble(0), 0.0 );
		assertEquals( 1.0, ds.getLabels().getRow(1).getDouble(1), 0.0 );
		
		assertEquals( false, iterator.hasNext() );
		
		
	}

}
