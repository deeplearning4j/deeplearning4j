/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.cli.records.reader;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.NoSuchElementException;

import org.datavec.api.conf.Configuration;
import org.datavec.api.formats.input.InputFormat;
import org.datavec.api.formats.output.OutputFormat;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.cli.subcommands.Vectorize;
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
		
		
		String datasetInputPath = (String) vecCommand.configProps.get("datavec.input.directory");
		
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
		
		
		String datasetInputPath = (String) vecCommand.configProps.get("datavec.input.directory");
		
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
