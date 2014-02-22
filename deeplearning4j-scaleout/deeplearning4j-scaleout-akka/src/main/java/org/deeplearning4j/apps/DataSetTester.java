package org.deeplearning4j.apps;

import java.io.FileNotFoundException;
import java.util.List;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.DeepLearningIOUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


public class DataSetTester {

	/**
	 * @param args
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException {
		CDBN c = new CDBN.Builder().buildEmpty();
		c.load(DeepLearningIOUtil.inputStreamFromPath(args[0]));
		DataSet data = DataSet.empty();
		data.load(DeepLearningIOUtil.inputStreamFromPath(args[1]));
		
		Evaluation eval = new Evaluation();
		
		int batch = Integer.parseInt(args[2]);
		List<DataSet> batches = data.dataSetBatches(batch);

		for(int i = 0; i < batches.size(); i++) {
			DataSet d = batches.get(i);
			DoubleMatrix first = d.getFirst();
			first = MatrixUtil.normalizeByColumnSums(first);
			DoubleMatrix second = d.getSecond();
			DoubleMatrix predicted = c.predict(first);
			eval.eval(second, predicted);
		}
		
		
		System.out.println(eval.stats());
		
	}

}
