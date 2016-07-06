/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.cli.subcommands;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Enumeration;
import java.util.Properties;

import org.canova.api.conf.Configuration;
import org.canova.api.exceptions.CanovaException;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.formats.output.OutputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.cli.vectorization.VectorizationEngine;
import org.canova.image.recordreader.ImageRecordReader;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Vectorize Command.
 * Based on an input and output format
 * transforms data
 * 
 * 
 * Design Notes
 * 		-	current version has some artifacts from previous iterations; we're not quite ready to have a unified set of interfaces for the 
 * 			vectorization pipelines
 * 		-	we've still yet to add the timeseries class of pipelines: { timeseries, audio, video }
 * 		-	we've still yet to add the parallelization mechanics to support yarn/spark
 * 		-	after we add the timeseries and parallelization mechaincs, we can refactor into a new inheritance setup
 * 
 * Label / RecordReader semantics
 * 		-	the record reader system is designed like the Hadoop RR system; it doenst know anything about contents of files
 * 		-	down the road we may want to change to a system where the label is treated specifically separately from the raw Collection<Writable> vector
 * 			-	this would be a cleaner design, but in v1 (w the impending book) we dont have time for this semantics change
 * 		-	it might be pragmatic to use a flexible vector schema, ala CSV's vector schema system, for other data types. not sure yet. 
 * 
 * 		-	Current breakdown for each pipeline
 * 			n.	Type:	Input Format			>	Label Mechanics in Collection<Writable>			// notes
 * 			1.	Image: 	ImageInputFormat		> 	{ [array of doubles], directoryLabelID }		// image data, then the directory indexed as an ID int
 * 			2.	CSV:	LineInputFormat			>	{ string }										// label is defined in vector schema
 * 			3.	Text:	(cli's) TextInputFormat	>	{ string, dirLabelString }						// label is second string in Collection<Writable>
 * 			4.	MNIST:	MnistInputFormat		>	{ [array of doubles], classIndexID }			// image data, then the class indexed as an ID int
 * 
 * 
 * 
 *
 * @author Josh Patterson
 * @author Adam Gibson
 */
public class Vectorize implements SubCommand {

    private static final Logger log = LoggerFactory.getLogger(Vectorize.class);

    public static final String OUTPUT_FILENAME_KEY = "canova.output.directory";
    public static final String INPUT_FORMAT = "canova.input.format";
    public static final String DEFAULT_INPUT_FORMAT_CLASSNAME = "org.canova.api.formats.input.impl.LineInputFormat";
    public static final String OUTPUT_FORMAT = "canova.output.format";
    public static final String DEFAULT_OUTPUT_FORMAT_CLASSNAME = "org.canova.api.formats.output.impl.SVMLightOutputFormat";

    public static final String VECTORIZATION_ENGINE = "canova.input.vectorization.engine";
    public static final String DEFAULT_VECTORIZATION_ENGINE_CLASSNAME = "org.canova.cli.csv.vectorization.CSVVectorizationEngine";

    public static final String NORMALIZE_DATA_FLAG = "canova.input.vectorization.normalize";
    public static final String SHUFFLE_DATA_FLAG = "canova.output.shuffle";
    public static final String PRINT_STATS_FLAG = "canova.input.statistics.debug.print";
    
    protected String[] args;

    public boolean validCommandLineParameters = true;

    @Option(name = "-conf", usage = "Sets a configuration file to drive the vectorization process")
    public String configurationFile = "";

    public Properties configProps = null;
    public String outputVectorFilename = "";
    public boolean normalizeData = true;

//    private CSVInputSchema inputSchema = null;
//    private CSVVectorizationEngine vectorizer = null;

    public Vectorize() {

    }
/*
    // this picks up the input schema file from the properties file and loads it
    private void loadInputSchemaFile() throws Exception {
        String schemaFilePath = (String) this.configProps.get("input.vector.schema");
        this.inputSchema = new CSVInputSchema();
        this.inputSchema.parseSchemaFile(schemaFilePath);
        this.vectorizer = new CSVVectorizationEngine();
    }
*/
    // picked up in the command line parser flags (-conf=<foo.txt>)
    public void loadConfigFile() throws IOException {

        this.configProps = new Properties();

        //Properties prop = new Properties();
        try (InputStream in = new FileInputStream(this.configurationFile)) {
            this.configProps.load(in);
        }

        if (null != this.configProps.get(NORMALIZE_DATA_FLAG)) {
        	String normalizeValue = (String) this.configProps.get(NORMALIZE_DATA_FLAG);
        	if ("false".equals(normalizeValue)) {
        		this.normalizeData = false;
        	}
        }

        if (null == this.configProps.get(OUTPUT_FILENAME_KEY)) {
            SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
            this.outputVectorFilename = "/tmp/canova_vectors_" + dateFormat.format(new Date()) + ".txt";
        } else {

            // what if its only a directory?

            this.outputVectorFilename = (String) this.configProps.get(OUTPUT_FILENAME_KEY);

            if (!(new File(this.outputVectorFilename).exists())) {

                // file path does not exist

                File yourFile = new File(this.outputVectorFilename);
                
                //File targetFile = new File("foo/bar/phleem.css");
                File parent = yourFile.getParentFile();
                if(!parent.exists() && !parent.mkdirs()){
                    throw new IllegalStateException("Couldn't create dir: " + parent);
                }                
                
                if (!yourFile.exists()) {
                    yourFile.createNewFile();
                }

            } else {
                if (new File(this.outputVectorFilename).isDirectory()) {
                    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
                    //File file = new File(dateFormat.format(date) + ".tsv") ;
                    this.outputVectorFilename += "/canova_vectors_" + dateFormat.format(new Date()) + ".txt";
                } else {
                    // if a file already exists
                    (new File(this.outputVectorFilename)).delete();
                    System.out.println("File path already exists, deleting the old file before proceeding...");
                }
            }
        }
    }

    /**
     * Dont change print stuff, its part of application console output UI
     * 
     */
    public void debugLoadedConfProperties() {
        Properties props = this.configProps; //System.getProperties();
        Enumeration e = props.propertyNames();

        System.out.println("\n--- Start Canova Configuration ---");

        while (e.hasMoreElements()) {
            String key = (String) e.nextElement();
            System.out.println(key + " -- " + props.getProperty(key));
        }

        System.out.println("---End Canova Configuration ---\n");
    }
    
    public static void printUsage() {
    	
    	System.out.println( "Canova: Vectorization Engine" );
    	System.out.println( "" );
    	System.out.println( "\tUsage:" );
    	System.out.println( "\t\tcanova vectorize -conf <conf_file>" );
    	System.out.println( "" );
    	System.out.println( "\tConfiguration File:" );
    	System.out.println( "\t\tContains a list of property entries that describe the vectorization process" );
    	System.out.println( "" );
    	System.out.println( "\tExample:" );
    	System.out.println( "\t\tcanova vectorize -conf /tmp/iris_conf.txt " );
    	
    	
    }


    // 1. load conf file
    // 2, load schema file
    // 3. transform csv -> output format
    public void execute() throws Exception  {
    	
    	if ("".equals(this.configurationFile)) {
    		printUsage();
    		return;
    	}

    	//System.out.println( "Vectorize > execute() [ START ]");
    	
        if (!this.validCommandLineParameters) {
            log.error("Vectorize function is not configured properly, stopping.");
            return;
        }

        //boolean schemaLoaded;
        // load stuff (conf, schema) --> CSVInputSchema

        this.loadConfigFile();

        if (null != this.configProps.get("canova.conf.print")) {
            String print = (String) this.configProps.get("canova.conf.print");
            if ("true".equals(print.trim().toLowerCase())) {
                this.debugLoadedConfProperties();
            }
        }



        // collect dataset statistics --> CSVInputSchema

        // [ first dataset pass ]
        // for each row in CSV Dataset

        String datasetInputPath = (String) this.configProps.get("canova.input.directory");
        String inputDataType = (String)this.configProps.get("canova.input.data.type");
        
        if ( null == inputDataType ) {
        	
        	// yeah cant do this, kick out
        	throw new IllegalStateException("Can't operate without input.data.type being set in the configuration file.");
        	
        }

        // ###########
        // THIS is where we end general vectorization and hit pipeline specific stuff
        // ###########
        Configuration conf = new Configuration();
        conf.set( OutputFormat.OUTPUT_PATH, this.outputVectorFilename );
        // hard set this on for images for now
        conf.setBoolean( ImageRecordReader.APPEND_LABEL, true);

        
        File inputFile = new File(datasetInputPath);
        InputSplit split = new FileSplit(inputFile);
        InputFormat inputFormat = this.createInputFormat();
        
        
        
        //System.out.println( "input file: " + datasetInputPath );

        RecordReader reader = inputFormat.createReader(split, conf);


        VectorizationEngine engine = this.createVectorizationEngine();
        engine.initialize(split, inputFormat, this.createOutputFormat(), reader, this.createOutputFormat().createWriter(conf), this.configProps, this.outputVectorFilename, conf );
        
        boolean vectorizationComplete = true;
        String failureString = "";
        
        try {
        	engine.execute();
        } catch (CanovaException ce) {
        	vectorizationComplete = false;
        	failureString = ce.toString();
        }
        
        if (!vectorizationComplete) {
        	
        	System.out.println( "Vectorization failed due to: \n" + failureString );
        	
        } else {
        	System.out.println( "Output vectors written to: " + this.outputVectorFilename );
        }
    }


    /**
     * @param args arguments for command
     */
    public Vectorize(String[] args) {

        this.args = args;
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            this.validCommandLineParameters = false;
            parser.printUsage(System.err);
            log.error("Unable to parse args", e);
        }

    }


    /**
     * Creates an input format
     *
     * @return
     */
    public InputFormat createInputFormat() {

        String clazz = (String) this.configProps.get(INPUT_FORMAT);

        if (null == clazz) {
            clazz = DEFAULT_INPUT_FORMAT_CLASSNAME;
        }

        try {
            Class<? extends InputFormat> inputFormatClazz = (Class<? extends InputFormat>) Class.forName(clazz);
            return inputFormatClazz.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }


    public OutputFormat createOutputFormat() {
        //String clazz = conf.get( OUTPUT_FORMAT, DEFAULT_OUTPUT_FORMAT_CLASSNAME );
        //System.out.println( "> Loading Output Format: " + (String) this.configProps.get( OUTPUT_FORMAT ) );
        String clazz = (String) this.configProps.get(OUTPUT_FORMAT);
        if (null == clazz) {
            clazz = DEFAULT_OUTPUT_FORMAT_CLASSNAME;
        }

        try {
            Class<? extends OutputFormat> outputFormatClazz = (Class<? extends OutputFormat>) Class.forName(clazz);
            return outputFormatClazz.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
    
    
    

    /**
     * Creates an input format
     *
     * @return {@link VectorizationEngine}
     */
    public VectorizationEngine createVectorizationEngine() {

    	String clazz = DEFAULT_VECTORIZATION_ENGINE_CLASSNAME; //(String) this.configProps.get( VECTORIZATION_ENGINE );
    	
    	// so this quick lookup is not the coolest way to do this, but for now we'll do it
    	
    	String inputDataType = (String)this.configProps.get("canova.input.data.type");

        switch (inputDataType) {
            case "csv":
                clazz = "org.canova.cli.vectorization.CSVVectorizationEngine";
                break;
            case "text":
                clazz = "org.canova.cli.vectorization.TextVectorizationEngine";
                break;
            case "audio":
                clazz = "org.canova.cli.vectorization.AudioVectorizationEngine";
                break;
            case "image":
                clazz = "org.canova.cli.vectorization.ImageVectorizationEngine";
                break;
            case "video":
                clazz = "org.canova.cli.vectorization.VideoVectorizationEngine";
                break;
            default:
                // stick to default --- should blow up (?)
                throw new IllegalArgumentException("Invalid input Data Type" + inputDataType);
        }
    	
        
/*
        if (null == clazz) {
            clazz = DEFAULT_VECTORIZATION_ENGINE_CLASSNAME;
        }
*/
    	
    //	log.debug("Running the " + clazz + " vectorization engine.");
    	
        try {
            Class<? extends VectorizationEngine> vecEngineClazz = (Class<? extends VectorizationEngine>) Class.forName(clazz);
            return vecEngineClazz.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }    
    
    

}