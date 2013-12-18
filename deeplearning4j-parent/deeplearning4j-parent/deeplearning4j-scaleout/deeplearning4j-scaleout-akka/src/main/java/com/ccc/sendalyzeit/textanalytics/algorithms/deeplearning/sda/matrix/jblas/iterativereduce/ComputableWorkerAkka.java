package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce;

import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.ComputableWorkerMatrix;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;

public class ComputableWorkerAkka extends ComputableWorkerMatrix implements DeepLearningConfigurable {

	private BaseMultiLayerNetwork network;
	private DoubleMatrix combinedInput;
	int fineTuneEpochs;
	int preTrainEpochs;
	int N;
	int[] hiddenLayerSizes;
	int numOuts;
	int numIns;
	int numHiddenNeurons;
	long seed;
	double learningRate;
	double corruptionLevel;
	int[] rows;
	private boolean iterationComplete;
	private int currEpoch;
	private DoubleMatrix outcomes;
	
	
	public ComputableWorkerAkka(DoubleMatrix whole,DoubleMatrix outcomes,int[] rows) {
		combinedInput = whole.getRows(rows);
		this.rows = rows;
		this.outcomes = outcomes.getRows(rows);
	}
	
	@Override
	public UpdateableMatrix compute(List<UpdateableMatrix> records) {
		return compute();
	}

	@Override
	public UpdateableMatrix compute() {
		network.trainNetwork(combinedInput, outcomes, new Object[]{});
		return new UpdateableMatrix(network);
	}

	@Override
	public boolean incrementIteration() {
		currEpoch++;
		return false;
	}

	@Override
	public void setup(Conf conf) {
	    N = conf.getInt(ROWS);
		hiddenLayerSizes = conf.getIntsWithSeparator(LAYER_SIZES, ",");
		numOuts = conf.getInt(OUT);
		numIns = conf.getInt(N_IN);
		numHiddenNeurons = conf.getIntsWithSeparator(LAYER_SIZES, ",").length;
		seed = conf.getLong(SEED);
		RandomGenerator rng = new MersenneTwister(conf.getLong(SEED));
		network = new BaseMultiLayerNetwork.Builder<>()
				.numberOfInputs(numIns).numberOfOutPuts(numOuts)
				.hiddenLayerSizes(hiddenLayerSizes).withRng(rng)
				.withClazz(conf.getClazz(CLASS)).build();
		learningRate = conf.getDouble(LEARNING_RATE);
		preTrainEpochs = conf.getInt(PRE_TRAIN_EPOCHS);
		fineTuneEpochs = conf.getInt(FINE_TUNE_EPOCHS);
		corruptionLevel = conf.getDouble(CORRUPTION_LEVEL);
		
	}


}
