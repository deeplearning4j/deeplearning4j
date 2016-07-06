package org.canova.cli.records.reader;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.NoSuchElementException;

import org.canova.api.conf.Configuration;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.formats.output.OutputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.cli.subcommands.Vectorize;
import org.junit.Test;

public class TestLineRecordReader {

	@Test
	public void test() throws IOException, InterruptedException {

		String[] args = { "-conf", "src/test/resources/text/DemoTextFiles/conf/text_vectorization_conf_unit_test.txt" };		
		Vectorize vecCommand = new Vectorize( args );
		vecCommand.loadConfigFile();
//		vecCommand.execute();
		
        Configuration conf = new Configuration();
        conf.set( OutputFormat.OUTPUT_PATH, "" );
		
		
		String datasetInputPath = (String) vecCommand.configProps.get("canova.input.directory");
		
		InputFormat inputformat = vecCommand.createInputFormat();
		
		//RecordReader rr = inputformat.
		
        File inputFile = new File( datasetInputPath );
        InputSplit split = new FileSplit(inputFile);
        //InputFormat inputFormat = this.createInputFormat();
        
        
        
        System.out.println( "input file: " + datasetInputPath );

        RecordReader reader = inputformat.createReader(split, conf );
        
        int count = 0;
        while (reader.hasNext()) {
        	
        	count++;
        	reader.next();
        	
        }
        
        assertEquals( 4, count );
				
	
	
	}

	@Test
	public void testMultiLabel() throws IOException, InterruptedException {

		String[] args = { "-conf", "src/test/resources/text/Tweets/conf/tweet_conf.txt" };		
		Vectorize vecCommand = new Vectorize( args );
		vecCommand.loadConfigFile();
//		vecCommand.execute();
		
        Configuration conf = new Configuration();
        conf.set( OutputFormat.OUTPUT_PATH, "" );
		
		
		String datasetInputPath = (String) vecCommand.configProps.get("canova.input.directory");
		
		InputFormat inputformat = vecCommand.createInputFormat();
		
		//RecordReader rr = inputformat.
		
        File inputFile = new File( datasetInputPath );
        InputSplit split = new FileSplit(inputFile);
        //InputFormat inputFormat = this.createInputFormat();
        
        
        
        System.out.println( "input file: " + datasetInputPath );

        RecordReader reader = inputformat.createReader(split, conf );
        
        int count = 0;
        while (reader.hasNext()) {
        	
       // 	System.out.println( "count: " + count );
        	
        	count++;
        	try {
        		reader.next();
        	} catch (NoSuchElementException e) {
        		
        	}
        	
        }
        
        assertEquals( 15, count );
				
	
	
	}	
	
}
