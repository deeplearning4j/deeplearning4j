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

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Enumeration;
import java.util.Properties;

import org.apache.commons.io.FileUtils;
import org.canova.api.formats.input.InputFormat;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.api.LayerFactory;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
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

    private static Logger log = LoggerFactory.getLogger(Train.class);


    // NOTE: disabled this setup for now for development purposes

    @Option(name = "-input", usage = "input data",aliases = "-i", required = true)
    private String input = "input.txt";


    @Option(name = "-output", usage = "location for saving model", aliases = "-o")
    private String outputDirectory = "output.txt";
    @Option(name = "-model",usage = "location for configuration of model",aliases = "-m")
    private String modelPath;
    @Option(name = "-type",usage = "type of network (layer or multi layer)")
    private String type = "multi";

    @Option(name = "-runtime", usage = "runtime- local, Hadoop, Spark, etc.", aliases = "-r", required = false)
    private String runtime = "local";

    @Option(name = "-properties", usage = "configuration for distributed systems", aliases = "-p", required = false)
    private String properties;
    @Option(name = "-savemode",usage = "output: (binary | txt)")
    private String saveMode = "txt";
    @Option(name = "-verbose",usage = "verbose(true | false)",aliases  = "-v")
    private boolean verbose = false;



    public Train() {
        this(new String[1]);
    }

    public Train(String[] args) {
        super(args);
    }

    /**
     * TODO:
     * 		-	lots of things to do here
     * 		-	runtime: if we're running on a cluster, then we have a different workflow / tracking setup
     *
     *
     */
    @Override
    public void execute() {
        try {
            loadConfigFile();
        } catch (Exception e) {
            e.printStackTrace();
        }

        if ("hadoop".equals(this.runtime.trim().toLowerCase()))
            this.execOnHadoop();

        else if ("spark".equals(this.runtime.trim().toLowerCase()))
            this.execOnSpark();

        else

            this.execLocal();



    }

    /**
     * Execute local training
     */
    public void execLocal() {
        log.warn( "[dl4j] - executing local ... " );
        log.warn( "using training input: " + this.input);

        File inputFile = new File(this.input);
        InputSplit split = new FileSplit( inputFile );
        InputFormat inputFormat = this.createInputFormat();

        RecordReader reader = null;

        try {
            reader = inputFormat.createReader(split);
        } catch (Exception e) {
            e.printStackTrace();
        }

        if(type.equals("multi")) {
            try {
                MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(modelPath)));
                FeedForwardLayer outputLayer = (FeedForwardLayer) conf.getConf(conf.getConfs().size() - 1).getLayer();

                DataSetIterator iter = new RecordReaderDataSetIterator( reader ,1,-1, outputLayer.getNOut());

                MultiLayerNetwork network = new MultiLayerNetwork(conf);
                if(verbose) {
                    network.init();
                    network.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));
                }
                network.fit(iter);
                if(saveMode.equals("binary")) {
                    BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(this.outputDirectory + File.separator + "outputmodel.bin"));
                    DataOutputStream dos = new DataOutputStream(bos);
                    Nd4j.write(network.params(),dos);
                }
                else {
                    Nd4j.writeTxt(network.params(),outputDirectory + File.separator + "outputmodel.txt",",");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else {
            try {
                NeuralNetConfiguration conf = NeuralNetConfiguration.fromJson(FileUtils.readFileToString(new File(modelPath)));
                LayerFactory factory = LayerFactories.getFactory(conf);
                int numParams = LayerFactories.getFactory(conf).initializer().numParams(conf,true);
                INDArray params = Nd4j.create(1, numParams);
                Layer l = factory.create(conf, null, 0, params);
                DataSetIterator iter = new RecordReaderDataSetIterator( reader , 1);
                while(iter.hasNext()) {
                    l.fit(iter.next().getFeatureMatrix());
                }

                if(saveMode.equals("binary")) {
                    BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(this.outputDirectory));
                    DataOutputStream dos = new DataOutputStream(bos);
                    Nd4j.write(l.params(),dos);
                }
                else {
                    Nd4j.writeTxt(l.params(),outputDirectory,",");
                }

            } catch (IOException e) {
                e.printStackTrace();
            }
        }


    }

    public void execOnSpark() {
        log.warn( "DL4J: Execution on spark from CLI not yet supported" );
    }

    public void execOnHadoop() {
        log.warn( "DL4J: Execution on hadoop from CLI not yet supported" );
    }

    /**
     * Create an input format
     * @return the input format to be created
     */
    public InputFormat createInputFormat() {
       if(configProps == null)
           try {
               loadConfigFile();
           } catch (Exception e) {
               e.printStackTrace();
           }
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


    public void loadConfigFile() throws Exception {

        this.configProps = new Properties();

        InputStream in = null;
        try {
            in = new FileInputStream(this.configurationFile);
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



        // get runtime - EXECUTION_RUNTIME_MODE_KEY
        if (this.configProps.get( EXECUTION_RUNTIME_MODE_KEY ) != null)
            this.runtime = (String) this.configProps.get(EXECUTION_RUNTIME_MODE_KEY);

        else
            this.runtime = EXECUTION_RUNTIME_MODE_DEFAULT;

        // get output directory
        if (null != this.configProps.get( OUTPUT_FILENAME_KEY ))
            this.outputDirectory = (String) this.configProps.get(OUTPUT_FILENAME_KEY);

        else
            // default
            this.outputDirectory = "/tmp/dl4j_model_default.txt";
        //throw new Exception("no output location!");



        // get input data

        if ( null != this.configProps.get( INPUT_DATA_FILENAME_KEY ))
            this.input = (String) this.configProps.get(INPUT_DATA_FILENAME_KEY);

        else
            throw new RuntimeException("no input file to train on!");

    }
}
