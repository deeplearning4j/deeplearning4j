package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce;

import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;

public interface DeepLearningConfigurable {
	/*  A csv of integers: the rows represented as part of a worker for a submatrix */
	public final static String ROWS = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.rows";
	/*  A csv of integers : the hidden layer sizes for the autoencoders*/
	public final static String LAYER_SIZES = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.layersizes"	;
	/* An integer the number of outputs*/
	public final static String OUT = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.out";

	/* int: Number of hidden layers */
	public final static String N_IN = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.hiddenlayers";

	/* A long */
	public final static String SEED = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.seed";
	/* A double: the starting learning rate for training */
	public final static String LEARNING_RATE = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.learningrate";
	/* The corruption level: the percent of inputs to be set to zero */
	public final static String CORRUPTION_LEVEL = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.corruptionlevel";
	/* The number of epochs to train on */
	public final static String FINE_TUNE_EPOCHS = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.epochs";
	/* The number of epochs to train on */
	public final static String PRE_TRAIN_EPOCHS = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.epochs";
	/* Input split: integer */
	public final static String SPLIT = "com.ccc.sendalyzeit.textanalytics.sda.matrix.jblas.split";
	/* Class to load for the base neural network*/
	public final static String CLASS = "com.ccc.sendalyzeit.textanalytics.class";
	
    void setup(Conf conf);
}
