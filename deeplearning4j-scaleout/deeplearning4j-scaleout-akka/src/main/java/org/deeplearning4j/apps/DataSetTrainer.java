package org.deeplearning4j.apps;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.List;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.nn.activation.HardTanh;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


public class DataSetTrainer {

	/**
	 * @param args
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException {
		String filePath = args[0];
		int batch = Integer.parseInt(args[1]);

		DataSet data = DataSet.empty();
		data.load(new BufferedInputStream(new FileInputStream(new File(filePath))));

		List<DataSet> batches = data.dataSetBatches(batch);

		int in = batches.get(0).getFirst().columns;
		int out = batches.get(0).numOutcomes();
		CDBN c = new CDBN.Builder().useRegularization(false)
				.numberOfInputs(in)
				.hiddenLayerSizes(new int[]{in / 2, in/ 4,in / 8})
				.withActivation(new HardTanh()).numberOfOutPuts(out)
				.build();

		for(int i = 0; i < batches.size(); i++) {
			DataSet d = batches.get(i);
			DoubleMatrix first = d.getFirst();
			first = MatrixUtil.normalizeByColumnSums(first);
			DoubleMatrix second = d.getSecond();
			c.pretrain(first, 1, 0.01, 1000);
			c.finetune(second, 0.01, 1000);
		}

		c.write(new BufferedOutputStream(new FileOutputStream(new File("nn-model.bin"))));

		/*ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);

		Conf c = new Conf();
		c.setnIn(batches.get(0).getFirst().columns);
		c.setnOut(batches.get(0).numOutcomes());
		c.setLayerSizes(new int[]{c.getnIn() / 2, c.getnIn() / 4,c.getnIn() / 8});
		c.setMultiLayerClazz(CDBN.class);
		c.setFunction(new HardTanh());
		c.setUseRegularization(false);
		c.setDeepLearningParams(Conf.getDefaultRbmParams());
		c.setSplit(10);
		runner.setup(c);
		runner.train();*/



	}

}
