package org.deeplearning4j.sda;


import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.junit.Before;
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
	public void testOutput() throws Exception {
        MnistDataFetcher fetcher = new MnistDataFetcher(true);
        fetcher.fetch(100);
        DataSet d = fetcher.next();
        d.filterAndStrip(new int[]{0, 1});
        log.info("Training on " + d.numExamples());
        StopWatch watch = new StopWatch();

        log.info("Data set " + d);

        StackedDenoisingAutoEncoder stackedDenoisingAutoEncoder = new StackedDenoisingAutoEncoder.Builder()
                .hiddenLayerSizes(new int[]{500,250,100})
                .withMomentum(0.5)
                .numberOfInputs(784)
                .numberOfOutPuts(fetcher.totalOutcomes())
                .build();

        watch.start();

        stackedDenoisingAutoEncoder.pretrain(d.getFirst(), 1, 1e-2, 300);
        stackedDenoisingAutoEncoder.finetune(d.getSecond(), 1e-2, 100);

        watch.stop();

        log.info("Took " + watch.getTime());
        Evaluation eval = new Evaluation();
        DoubleMatrix predict = stackedDenoisingAutoEncoder.output(d.getFirst());
        eval.eval(d.getSecond(), predict);
        log.info(eval.stats());


	}








}
