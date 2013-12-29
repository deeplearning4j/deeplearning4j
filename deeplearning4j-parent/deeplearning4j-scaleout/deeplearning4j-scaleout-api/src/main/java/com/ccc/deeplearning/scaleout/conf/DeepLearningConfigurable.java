package com.ccc.deeplearning.scaleout.conf;

import com.ccc.deeplearning.scaleout.conf.Conf;

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
	/* Network implementation specific parameters */
	public final 
	static String PARAMS = "com.ccc.sendalyzeit.textanalytics.params";
	/* L2 regularization constant */
	public final static String L2 = "com.ccc.deeplearning.l2";
	/* Momentum */
	public final static String MOMENTUM = "com.ccc.deeplearning.momentum";
	
	public final static String PARAM_ALGORITHM = "algorithm";
	public final static String PARAM_SDA = "sda";
	public final static String PARAM_CDBN = "cdbn";
	public final static String PARAM_DBN = "dbn";
	public final static String PARAM_K = "k";
	public final static String PARAM_EPOCHS = "epochs";
	public final static String PARAM_FINETUNE_LR = "finetunelr";
	public final static String PARAM_FINETUNE_EPOCHS = "finetunepochs";
	public final static String PARAM_CORRUPTION_LEVEL = "corruptionlevel";
	public final static String PARAM_LEARNING_RATE = "lr";

	
    void setup(Conf conf);
}
