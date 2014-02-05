package com.ccc.deeplearning.apps;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.dbn.CDBN;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.util.DeepLearningIOUtil;
import com.ccc.deeplearning.util.MatrixUtil;

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
