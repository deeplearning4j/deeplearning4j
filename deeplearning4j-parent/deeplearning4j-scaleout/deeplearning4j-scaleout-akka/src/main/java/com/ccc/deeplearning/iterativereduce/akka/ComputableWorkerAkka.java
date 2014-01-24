package com.ccc.deeplearning.iterativereduce.akka;

import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.nn.activation.ActivationFunction;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.iterativereduce.multi.ComputableWorkerImpl;
import com.ccc.deeplearning.scaleout.iterativereduce.multi.UpdateableImpl;

public class ComputableWorkerAkka extends ComputableWorkerImpl implements DeepLearningConfigurable {

	private BaseMultiLayerNetwork network;
	private DoubleMatrix combinedInput;
	int fineTuneEpochs;
	int preTrainEpochs;
	boolean useRegularization;
	int[] hiddenLayerSizes;
	int numOuts;
	int numIns;
	double momentum = 0.0;
	int numHiddenNeurons;
	long seed;
	double learningRate;
	double corruptionLevel;
	ActivationFunction activation;
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
		if(conf.containsKey(USE_REGULARIZATION))
			this.useRegularization = conf.getBoolean(USE_REGULARIZATION);
		if(conf.containsKey(MOMENTUM))
			momentum = conf.getDouble(MOMENTUM);
		if(conf.containsKey(ACTIVATION))
		   this.activation = conf.getFunction(ACTIVATION);
		
		RandomGenerator rng = new MersenneTwister(conf.getLong(SEED));
		network = new BaseMultiLayerNetwork.Builder<>()
				.numberOfInputs(numIns).numberOfOutPuts(numOuts)
				.withActivation(activation)
				.hiddenLayerSizes(hiddenLayerSizes).withRng(rng)
				.useRegularization(useRegularization).withMomentum(momentum)
				.withClazz(conf.getClazz(CLASS)).build();
		
		
		learningRate = conf.getDouble(LEARNING_RATE);
		preTrainEpochs = conf.getInt(PRE_TRAIN_EPOCHS);
		fineTuneEpochs = conf.getInt(FINE_TUNE_EPOCHS);
		corruptionLevel = conf.getDouble(CORRUPTION_LEVEL);
		extraParams = conf.loadParams(PARAMS);
	}


}
