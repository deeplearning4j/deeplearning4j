package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base;

import java.io.File;
import java.io.IOException;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.MnistManager;



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

			return new Pair<DoubleMatrix,DoubleMatrix>(asMatrix(imageExample),toOutcomeVector(man.readLabel(),10));
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

			return new Pair<DoubleMatrix,DoubleMatrix>(asMatrix(imageExample),toOutcomeVector(man.readLabel(),10));

		}


	}


	public DoubleMatrix toOutcomeVector(int index,int numOutcomes) {
		int[] nums = new int[numOutcomes];
		nums[index] = 1;
		return asMatrix(nums);
	}

	/* Return a column vector */
	private DoubleMatrix asMatrix(int[] nums) {
		DoubleMatrix ret = new DoubleMatrix(nums.length);
		for(int i = 0; i < ret.length; i++)
			ret.put(i,nums[i]);
		return ret.transpose();
	}

	private boolean isSquare(int[][] nums) {
		int firstLength = nums[0].length;
		for(int i = 0; i < nums.length; i++) {
			if(nums.length != firstLength)
				return false;
		}
		return true;
	}


	private DoubleMatrix asMatrix(int[][] nums) {
		if(!isSquare(nums))
			throw new IllegalStateException("WTF IS THIS");

		DoubleMatrix ret = new DoubleMatrix(nums.length,nums[0].length);

		for(int i = 0; i < nums.length; i++)
			for(int j = 0; j < nums[i].length; j++)
				ret.put(i,j,nums[i][j]);
		return ret;
	}




}
