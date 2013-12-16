package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base;

import java.io.File;
import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.MnistManager;
import com.ccc.sendalyzeit.textanalytics.util.ArrayUtil;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;



public abstract class DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(DeepLearningTest.class);

	public Pair<DoubleMatrix,DoubleMatrix> getIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> pair = IrisUtils.loadIris();
		return pair;
	}
	/**
	 * Gets an mnist example as an input, label pair.
	 * Keep in mind the return matrix for out come is a 1x1 matrix.
	 * If you need multiple labels, remember to convert to a zeros
	 * with 1 as index of the label for the output training vector.
	 * @param example the example to get
	 * @return the image,label pair
	 * @throws IOException
	 */
	public Pair<DoubleMatrix,DoubleMatrix> getMnistExample(int example) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		if(ensureExists.exists()) {
			MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
			man.setCurrent(example);
			int[][] image = man.readImage();
			int[] imageExample = new int[image.length * image[0].length];
			int linearCount = 0;
			for(int i = 0; i < image.length; i++)
				for(int j = 0; j < image[i].length; j++) {
					imageExample[linearCount++] = image[i][j];
				}

			return new Pair<DoubleMatrix,DoubleMatrix>(MatrixUtil.toMatrix(imageExample),MatrixUtil.toOutcomeVector(man.readLabel(),10));
		}
		else {
			MnistFetcher fetcher = new MnistFetcher();
			fetcher.downloadAndUntar();
			MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
			man.setCurrent(example);
			int[][] image = man.readImage();
			int[] imageExample = new int[image.length * image[0].length];
			int linearCount = 0;
			for(int i = 0; i < image.length; i++)
				for(int j = 0; j < image[i].length; j++) {
					imageExample[linearCount++] = image[i][j];
				}

			return new Pair<DoubleMatrix,DoubleMatrix>(MatrixUtil.toMatrix(imageExample),MatrixUtil.toOutcomeVector(man.readLabel(),10));

		}


	}


	/**
	 * Gets an mnist example as an input, label pair.
	 * Keep in mind the return matrix for out come is a 1x1 matrix.
	 * If you need multiple labels, remember to convert to a zeros
	 * with 1 as index of the label for the output training vector.
	 * @param example the example to get
	 * @param batchSize the batch size of examples to get
	 * @return the image,label pair
	 * @throws IOException
	 */
	public Pair<DoubleMatrix,DoubleMatrix> getMnistExampleBatch(int batchSize) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		if(!ensureExists.exists()) 
			new MnistFetcher().downloadAndUntar();
		MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
		
		int[][] image = man.readImage();
		int[] imageExample = ArrayUtil.flatten(image);
		int[][] examples = new int[batchSize][imageExample.length];
		int[][] outcomeMatrix = new int[batchSize][10];
		for(int i = 1; i < batchSize; i++) {
			man.setCurrent(i);
			int[] currExample = ArrayUtil.flatten(man.readImage());
			examples[i] = currExample;
			outcomeMatrix[i] = ArrayUtil.toOutcomeArray(man.readLabel(), 10);
		}



		return new Pair<DoubleMatrix,DoubleMatrix>(MatrixUtil.toMatrix(imageExample),MatrixUtil.toMatrix(outcomeMatrix));

	}


}





