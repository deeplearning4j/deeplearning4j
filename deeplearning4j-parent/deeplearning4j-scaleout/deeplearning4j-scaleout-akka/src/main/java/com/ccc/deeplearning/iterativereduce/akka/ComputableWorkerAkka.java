package com.ccc.deeplearning.iterativereduce.akka;

import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.iterativereduce.multi.ComputableWorkerImpl;
import com.ccc.deeplearning.scaleout.iterativereduce.multi.UpdateableImpl;

public class ComputableWorkerAkka extends ComputableWorkerImpl implements DeepLearningConfigurable {

	private BaseMultiLayerNetwork network;
	private DoubleMatrix combinedInput;
	int fineTuneEpochs;
	int preTrainEpochs;
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
	Object[] extraParams;
	
	public ComputableWorkerAkka(DoubleMatrix whole,DoubleMatrix outcomes,int[] rows) {
		combinedInput = whole.getRows(rows);
		this.rows = rows;
		this.outcomes = outcomes.getRows(rows);
	}
	
	@Override
	public UpdateableImpl compute(List<UpdateableImpl> records) {
		return compute();
	}

	@Override
	public UpdateableImpl compute() {
		network.trainNetwork(combinedInput, outcomes,extraParams);
		return new UpdateableImpl(network);
	}

	@Override
	public boolean incrementIteration() {
		currEpoch++;
		return false;
	}

	@Override
	public void setup(Conf conf) {
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
		extraParams = conf.loadParams(PARAMS);
	}


}
