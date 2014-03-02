package org.deeplearning4j.sda;
import static org.junit.Assert.*;


import java.io.IOException;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.base.IrisUtils;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.activation.HardTanh;
import org.deeplearning4j.sda.StackedDenoisingAutoEncoder;
import org.deeplearning4j.transformation.MultiplyScalar;
import org.deeplearning4j.transformation.ScalarMatrixTransform;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class SdaTest {

	double pretrain_lr = 0.001;
	double corruption_level = 0.8;
	int pretraining_epochs = 50;
	double finetune_lr = 0.001;
	int finetune_epochs = 50;
	int test_N = 2;
	RandomGenerator rng = new JDKRandomGenerator();
	private static Logger log = LoggerFactory.getLogger(SdaTest.class);
	int n_ins = 2;
	int n_outs = 2;
	int[] hidden_layer_sizes_arr = {300,200,100};
	int n_layers = hidden_layer_sizes_arr.length;

	int seed = 123;

	@Before
	public void init() {
		rng.setSeed(seed);

	}





	@Test
	public void testOutput() {
		DataSet xor = MatrixUtil.xorData(100, 200);


		StackedDenoisingAutoEncoder sda = new StackedDenoisingAutoEncoder.Builder()
		.hiddenLayerSizes(hidden_layer_sizes_arr)
		.numberOfInputs(xor.getFirst().columns).numberOfOutPuts(xor.getSecond().columns).renderWeights(0)
		.useRegularization(false).withMomentum(0.).withRng(rng).build();		


		DoubleMatrix x = xor.getFirst();

		sda.pretrain(x,pretrain_lr, corruption_level, pretraining_epochs);
		// finetune

		sda.finetune(xor.getSecond(),finetune_lr, finetune_epochs);
		log.info("OUTPUT TEST");
		DoubleMatrix predicted = sda.predict(x);

		Evaluation eval = new Evaluation();
		eval.eval(xor.getSecond(), predicted);
		log.info(eval.stats());


	}








}
