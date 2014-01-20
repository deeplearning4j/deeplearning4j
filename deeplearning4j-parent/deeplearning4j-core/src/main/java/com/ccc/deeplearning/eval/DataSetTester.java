package com.ccc.deeplearning.eval;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.dbn.DBN;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
/**
 * DataSet runner main class.
 * 
 * Basic idea is to feed it an algorithm, dataset, and the number of examples to use.
 * It will then print out f1 scores for each dataset.
 * 
 * Note that I need to add WAY more for tuning this yet as far as command line options go.
 * @author Adam Gibson
 *
 */
public class DataSetTester extends DeepLearningTest {

	private static int[] layers = new int[] {200,200,200};
	private String dataset;
	private String algorithm;
	private Integer numExamples;
	private static Logger log = LoggerFactory.getLogger(DataSetTester.class);
	
	public DataSetTester(String dataset, String algorithm, Integer numExamples) {
		super();
		this.dataset = dataset;
		this.algorithm = algorithm;
		this.numExamples = numExamples;
	}
	
	public DataSetTester(String dataset, String algorithm) {
		super();
		this.dataset = dataset;
		this.algorithm = algorithm;
	}


	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		String algorithm = args[0];
		String dataset = args[1];
        if(args.length > 2) {
        	int num = Integer.parseInt(args[2]);
        	DataSetTester test = new DataSetTester(dataset,algorithm,num);
        	test.run();
        	
        }
        else {
        	DataSetTester test = new DataSetTester(dataset,algorithm);
        	test.run();

        }
		
	}
	
	public void run() throws Exception {
		List<Pair<DoubleMatrix,DoubleMatrix>> dataset =  null;
		if(numExamples != null) 
			dataset = loadDataset(numExamples);
			
		else 
			dataset = loadDataset();
		
		BaseMultiLayerNetwork neuralNet = getNeuralNet(dataset);
		long start = System.currentTimeMillis();
		Evaluation e = new Evaluation();

		for(Pair<DoubleMatrix,DoubleMatrix> pair : dataset) {
			neuralNet.trainNetwork(pair.getFirst(), pair.getSecond(), getOtherParams());
			DoubleMatrix predicted = neuralNet.predict(pair.getFirst());
			e.eval(pair.getSecond(), predicted);
		}
		
		long end = System.currentTimeMillis();
		long diff = end - start;
		
		log.info("Ended in " + TimeUnit.MILLISECONDS.toSeconds(diff) + " seconds");
		
		log.info(e.stats());
		
	}
	
	private Object[] getOtherParams() {
		if(algorithm.equals("sda")) {
			return new Object[]{0.1,0.3,500,0.1,200};
		}
		else if(algorithm.equals("dbn") || algorithm.equals("cdbn")) {
			return new Object[]{1,0.1,500,0.1,200};

		}
		
		return null;
	}
	
	
	
	private BaseMultiLayerNetwork getNeuralNet(List<Pair<DoubleMatrix,DoubleMatrix>> dataset) {
		Pair<Integer,Integer> params = numInputsOutcomes(dataset);
		BaseMultiLayerNetwork ret = new BaseMultiLayerNetwork.Builder<>()
				.hiddenLayerSizes(layers).numberOfInputs(params.getFirst())
				.numberOfOutPuts(params.getSecond()).withRng(new MersenneTwister(123))
				.withClazz(algorithmForClass()).build();
		return ret;
		
	}
	
	
	private Class<? extends BaseMultiLayerNetwork> algorithmForClass() {
		if(algorithm.equals("sda"))
			return BaseMultiLayerNetwork.class;
		else if(algorithm.equals("cdbn"))
			return CDBN.class;
		else if(algorithm.equals("dbn"))
			return DBN.class;
		throw new IllegalStateException("No algorithm found");
	}
	
	private Pair<Integer,Integer> numInputsOutcomes(List<Pair<DoubleMatrix,DoubleMatrix>> list) {
		return numInputsOutcomes(list.get(0));
	}

	private Pair<Integer,Integer> numInputsOutcomes(Pair<DoubleMatrix,DoubleMatrix> pair) {
		int numInputs = pair.getFirst().columns;
		int numOutcomes = pair.getSecond().columns;
		return new Pair<>(numInputs,numOutcomes);
	}

	private  List<Pair<DoubleMatrix,DoubleMatrix>> loadDataset(int numExamples) throws Exception {
		if(dataset.equals("lfw")) {
			return getFirstFaces(numExamples);
		}

		else if(dataset.equals("iris")) {
			return Collections.singletonList(getIris());
		}
		else if(dataset.equals("mnist")) {
			return this.getMnistExampleBatches(1, numExamples);
		}

		return null;


	}

	private  List<Pair<DoubleMatrix,DoubleMatrix>> loadDataset() throws Exception {
		if(dataset.equals("lfw")) {
			return getFaces();
		}

		else if(dataset.equals("iris")) {
			return Collections.singletonList(getIris());
		}
		else if(dataset.equals("mnist")) {
			return this.getMnistExampleBatches(10, 6000);
		}

		return null;


	}

}
