package org.deeplearning4j.scaleout.conf;

import org.deeplearning4j.scaleout.conf.Conf;

public interface DeepLearningConfigurable {
	/*  A csv of integers: the rows represented as part of a worker for a submatrix */
	public final static String ROWS = "org.deeplearning4j.rows";
	/*  A csv of integers : the hidden layer sizes for the autoencoders*/
	public final static String LAYER_SIZES = "org.deeplearning4j.layersizes"	;
	/* An integer the number of outputs*/
	public final static String OUT = "org.deeplearning4j.out";

	/* int: Number of hidden layers */
	public final static String N_IN = "org.deeplearning4j.hiddenlayers";

	/* A long */
	public final static String SEED = "org.deeplearning4j.seed";
	/* A double: the starting learning rate for training */
	public final static String LEARNING_RATE = "org.deeplearning4j.learningrate";
	/* The corruption level: the percent of inputs to be applyTransformToDestination to zero */
	public final static String CORRUPTION_LEVEL = "org.deeplearning4j.corruptionlevel";
	/* The number of epochs to iterate on */
	public final static String FINE_TUNE_EPOCHS = "org.deeplearning4j.epochs";
	/* The number of epochs to iterate on */
	public final static String PRE_TRAIN_EPOCHS = "org.deeplearning4j.epochs";
	/* Input split: integer */
	public final static String SPLIT = "org.deeplearning4j.split";
	/* Class to load for the base neural network*/
	public final static String CLASS = "org.deeplearning4j.sendalyzeit.textanalytics.class";
	/* Network implementation specific parameters */
	public final 
	static String PARAMS = "org.deeplearning4j.sendalyzeit.textanalytics.params";
	/* L2 regularization constant */
	public final static String L2 = "org.deeplearning4j.l2";
	/* Momentum */
	public final static String MOMENTUM = "org.deeplearning4j.momentum";
	
	/* activation function */
	public final static String ACTIVATION = "org.deeplearning4j.activation";
	
	public final static String USE_REGULARIZATION = "org.deeplearning4j.reg";
	
	public final static String PARAM_ALGORITHM = "algorithm";
	public final static String PARAM_SDA = "sda";
	public final static String PARAM_CDBN = "cdbn";
	public final static String PARAM_DBN = "dbn";
	public final static String PARAM_RBM = "rbm";
	public final static String PARAM_CRBM = "crbm";
	public final static String PARAM_DA = "da";
	public final static String PARAM_K = "k";
	public final static String PARAM_EPOCHS = "epochs";
	public final static String PARAM_FINETUNE_LR = "finetunelr";
	public final static String PARAM_FINETUNE_EPOCHS = "finetunepochs";
	public final static String PARAM_CORRUPTION_LEVEL = "corruptionlevel";
	public final static String PARAM_LEARNING_RATE = "lr";
	/* Number of passes to do on the data applyTransformToDestination */
	public final static String NUM_PASSES = "org.deeplearning4j.numpasses";
	
    void setup(Conf conf);
}
