/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.cli.subcommands;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Enumeration;
import java.util.Properties;

import org.canova.api.formats.input.InputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.api.LayerFactory;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Subcommand for training model
 *
 * Options:
 *      Required:
 *          -input: input data file for model
 *          -model: json configuration for model
 *
 * @author sonali
 */
public class Train extends BaseSubCommand {


	public static final String EXECUTION_RUNTIME_MODE_KEY = "execution.runtime";
	public static final String EXECUTION_RUNTIME_MODE_DEFAULT = "local";

	public static final String OUTPUT_FILENAME_KEY = "output.directory";
	public static final String INPUT_DATA_FILENAME_KEY = "input.directory";

	public static final String INPUT_FORMAT_KEY = "input.format";
	public static final String DEFAULT_INPUT_FORMAT_CLASSNAME = "org.canova.api.formats.input.impl.SVMLightInputFormat";

	@Option(name = "-conf", usage = "configuration file for training", required = true )
	public String configurationFile = "";

	public Properties configProps = null;
	public String outputVectorFilename = "";
	
	private static Logger log = LoggerFactory.getLogger(Train.class);


	// NOTE: disabled this setup for now for development purposes

	@Option(name = "-input", usage = "input data",aliases = "-i", required = true)
	private String input = "input.txt";


	@Option(name = "-output", usage = "location for saving model", aliases = "-o")
	private String outputDirectory = "output.txt";

	@Option(name = "-runtime", usage = "runtime- local, Hadoop, Spark, etc.", aliases = "-r", required = false)
	private String runtime = "local";

	@Option(name = "-properties", usage = "configuration for distributed systems", aliases = "-p", required = false)
	private String properties;

	public Train(String[] args) {
		super(args);

		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			//this.validCommandLineParameters = false;
			parser.printUsage(System.err);
			//log.error("Unable to parse args", e);
		}


	}

	/**
	 * TODO:
	 * 		-	lots of things to do here
	 * 		-	runtime: if we're running on a cluster, then we have a different workflow / tracking setup
	 *
	 *
	 */
	@Override
	public void exec() {

		if ("hadoop".equals(this.runtime.trim().toLowerCase())) {

			this.execOnHadoop();

		} else if ("spark".equals(this.runtime.trim().toLowerCase())) {

			this.execOnSpark();

		} else {

			this.execLocal();

		}

	}

	public void execLocal() {

		log.warn( "[dl4j] - executing local ... " );
		log.warn( "using training input: " + this.input );

		File inputFile = new File( this.input );
		InputSplit split = new FileSplit( inputFile );
		InputFormat inputFormat = this.createInputFormat();

		RecordReader reader = null;

		try {
			reader = inputFormat.createReader(split);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//FileSplit csv = new FileSplit(new ClassPathResource("csv-example.csv").getFile());
		//recordReader.initialize(csv);
		DataSetIterator iter = new RecordReaderDataSetIterator( reader , 20 );
		DataSet next = iter.next();
		//assertEquals(34,next.numExamples());

		log.warn( "[dl4j:exec] examples in dataset: " + next.numExamples() );
    	/*
        LayerFactory layerFactory = LayerFactories.getFactory(OutputLayer.class);
    	
        OutputLayer l = layerFactory.create(conf, Arrays.<IterationListener>asList(new ScoreIterationListener(1)));
    	
        //DataSet next = iter.next();
        //SplitTestAndTrain trainTest = next.splitTestAndTrain(110);
        //trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain trainTest = next.splitTestAndTrain(0);
        trainTest.getTrain().normalizeZeroMeanZeroUnitVariance();
        
        l.fit( trainTest.getTrain() );        
        */
	}

	public void execOnSpark() {

		log.warn( "DL4J: Execution on spark from CLI not yet supported" );

	}

	public void execOnHadoop() {

		log.warn( "DL4J: Execution on hadoop from CLI not yet supported" );

	}

	public InputFormat createInputFormat() {

		//log.warn( "> Loading Input Format: " + (String) this.configProps.get( INPUT_FORMAT ) );

		String clazz = (String) this.configProps.get( INPUT_FORMAT_KEY );

		if ( null == clazz ) {
			clazz = DEFAULT_INPUT_FORMAT_CLASSNAME;
		}

		try {
			Class<? extends InputFormat> inputFormatClazz = (Class<? extends InputFormat>) Class.forName(clazz);
			return inputFormatClazz.newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}


	public void loadConfigFile() throws Exception, IOException {

		this.configProps = new Properties();

		//log.warn( "Loading Conf file: " + this.configurationFile );

		//Properties prop = new Properties();
		InputStream in = null;
		try {
			in = new FileInputStream( this.configurationFile );
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			this.configProps.load(in);
			in.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//this.debugLoadedConfProperties();

		// get runtime - EXECUTION_RUNTIME_MODE_KEY

		if (null != this.configProps.get( EXECUTION_RUNTIME_MODE_KEY )) {

			this.runtime = (String) this.configProps.get(EXECUTION_RUNTIME_MODE_KEY);

		} else {

			this.runtime = EXECUTION_RUNTIME_MODE_DEFAULT;

		}


		// get output directory

		if (null != this.configProps.get( OUTPUT_FILENAME_KEY )) {

			this.outputDirectory = (String) this.configProps.get(OUTPUT_FILENAME_KEY);

		} else {

			// default
			this.outputDirectory = "/tmp/dl4_model_default.txt";
			//throw new Exception("no output location!");

		}

		// get input data

		if ( null != this.configProps.get( INPUT_DATA_FILENAME_KEY )) {

			//log.warn( "\nLOADED INPUT SRC\n\n" );
			this.input = (String) this.configProps.get(INPUT_DATA_FILENAME_KEY);

		} else {

			// default
			//this.input = "/tmp/dl4_model_default.txt";
			throw new Exception("no input file to train on!");

		}		
			
	/*
		
		if (null == this.configProps.get( OUTPUT_FILENAME_KEY )) {
			
			Date date = new Date() ;
			SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss") ;
			this.outputVectorFilename = "/tmp/canova_vectors_" + dateFormat.format(date) + ".txt";
						
		} else {
			
			// what if its only a directory?
			
			this.outputVectorFilename = (String) this.configProps.get( OUTPUT_FILENAME_KEY );
			
			if ( (new File( this.outputVectorFilename ).exists()) == false ) {
				
				// file path does not exist
				
				File yourFile = new File( this.outputVectorFilename );
				if(!yourFile.exists()) {
				    yourFile.createNewFile();
				} 
				
			} else {
				
				if ( new File( this.outputVectorFilename ).isDirectory() ) {
					
					
					Date date = new Date() ;
					SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss") ;
					//File file = new File(dateFormat.format(date) + ".tsv") ;
					
					this.outputVectorFilename += "/canova_vectors_" + dateFormat.format(date) + ".txt";
					
					
				} else {
					
					// if a file that exists
					
					
					(new File( this.outputVectorFilename )).delete();
					
					log.warn( "File path already exists, deleting the old file before proceeding..." );
					
					
				}
				
				
			}
			*/
		//log.warn( "Writing vectorized output to: " + this.outputVectorFilename + "\n\n" );

		//}


	}


	public void debugLoadedConfProperties() {

		Properties props = this.configProps; //System.getProperties();
		Enumeration e = props.propertyNames();

		log.warn("\n-- DL4J Configuration --");

		while (e.hasMoreElements()) {
			String key = (String) e.nextElement();
			log.warn(key + " - " + props.getProperty(key));
		}

		log.warn("-- DL4J Configuration --\n");
	}



}
