package org.canova.hadoop.records.reader;

import static org.junit.Assert.*;

import java.io.IOException;

import org.junit.Test;



import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Counters.Counter;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RecordReader;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.TaskAttemptContext;
import org.junit.Test;

/**
 * Notes
 * 
 * https://linuxjunkies.wordpress.com/2011/11/21/a-hdfsclient-for-hadoop-using-the-native-java-api-a-tutorial/
 * 
 * 
 * 
 * Spark Notes on input formats
		When Spark reads a file from HDFS, it creates a single partition for a single input split. 
		Input split is set by the Hadoop InputFormat used to read this file. For instance, 
		if you use textFile() it would be TextInputFormat in Hadoop, 
		which would return you a single partition for a single block of 
		HDFS (but the split between partitions would be done on line split, not the exact block split), 
		unless you have a compressed text file. 
		In case of compressed file you would get a single partition for a single 
		file (as compressed text files are not splittable).
 * 
 * @author josh
 *
 */
public class TestBasicHDFS_Integration {

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


	/**
	 * generate splits for this run
	 * 
	 * @param input_path
	 * @param job
	 * @return
	 */
	private InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

		long block_size = localFs.getDefaultBlockSize();

		System.out.println("default block size: " + (block_size / 1024 / 1024) + "MB");

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
	 * Things we'd need:
	 * 		1. JobConf
	 *		2. some way to get input splits
	 * 
	 * @throws IOException
	 */
	@Test
	public void testParametersInputSplitSetup() throws IOException {
		
//		InputSplit genericSplit = null;
        
//		TaskAttemptContext context = null;
		
		// ---- this all needs to be done in
		JobConf job = new JobConf(defaultConf);
		
		// app.input.path

		String split_filename = "src/test/resources/records/reader/SVMLightRecordReaderInput/record_reader_input_test.txt";

		Path splitPath = new Path( split_filename );
		
		
		
		InputSplit[] splits = generateDebugSplits(splitPath, job);

		System.out.println("split count: " + splits.length);
		
		//RecordReader<LongWritable, Text> rr = new LineRecordReader(job, (FileSplit) splits[0]);
		
		TextInputFormat format = new TextInputFormat();
		format.configure(job);
		
		//Reporter reporter = new DummyReporter();
		
		RecordReader<LongWritable, Text> reader = null;
		  LongWritable key = new LongWritable();
		  Text value = new Text();
		  
		  final Reporter voidReporter = Reporter.NULL;		
		
		reader = format.getRecordReader(splits[0], job, voidReporter);

		//while (rr.)
		
		while (reader.getProgress() < 1.0) {
			
			boolean hasMore = reader.next(key, value);
			
			System.out.println( "line: "+ value.toString() );
			
		}
		
		reader.close();
		
	}
	
	@Test 
	public void testParameters_Alt() {
		
	//	TaskAttemptContext context = new TaskAttemptContext();
		
		//return new LineRecordReader(job, (FileSplit) genericSplit);
		
		
	}
	
}
