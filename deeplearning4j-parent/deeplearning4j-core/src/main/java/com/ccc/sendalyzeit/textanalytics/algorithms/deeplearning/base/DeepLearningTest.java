package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.MnistManager;
import com.ccc.sendalyzeit.textanalytics.util.ArrayUtil;
import com.ccc.sendalyzeit.textanalytics.util.MathUtils;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;



public abstract class DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(DeepLearningTest.class);

	public static Pair<DoubleMatrix,DoubleMatrix> getIris() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> pair = IrisUtils.loadIris();
		return pair;
	}
	public static Pair<DoubleMatrix,DoubleMatrix> getIris(int num) throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> pair = IrisUtils.loadIris(num);
		return pair;
	}
	
	
	
	
	/**
	 * LFW Dataset: pick first num faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
	public static Pair<DoubleMatrix,DoubleMatrix> getFaces(int num) throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getAllImagesAsMatrix(num);
	}
	
	
	/**
	 * LFW Dataset: pick all faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
	public static Pair<DoubleMatrix,DoubleMatrix> getFacesMatrix() throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getAllImagesAsMatrix();
	}
	
	
	
	/**
	 * LFW Dataset: pick first num faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
	public static List<Pair<DoubleMatrix,DoubleMatrix>> getFirstFaces(int num) throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getFirst(num);
	}
	
	
	/**
	 * LFW Dataset: pick all faces
	 * @param num
	 * @return
	 * @throws Exception
	 */
	public List<Pair<DoubleMatrix,DoubleMatrix>> getFaces() throws Exception {
		LFWLoader loader = new LFWLoader();
		loader.getIfNotExists();
		return loader.getImagesAsList();
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
	public static Pair<DoubleMatrix,DoubleMatrix> getMnistExample(int example) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		if(!ensureExists.exists())
			new MnistFetcher().downloadAndUntar();

		MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
		man.setCurrent(example);
		int[] imageExample = ArrayUtil.flatten(man.readImage());
		return new Pair<DoubleMatrix,DoubleMatrix>(MatrixUtil.toMatrix(imageExample).transpose(),MatrixUtil.toOutcomeVector(man.readLabel(),10));
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
	public List<Pair<DoubleMatrix,DoubleMatrix>> getMnistExampleBatches(int batchSize,int numBatches) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		List<Pair<DoubleMatrix,DoubleMatrix>> ret = new ArrayList<>();
		if(!ensureExists.exists()) 
			new MnistFetcher().downloadAndUntar();
		MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);

		int[][] image = man.readImage();
		int[] imageExample = ArrayUtil.flatten(image);

		for(int batch = 0; batch < numBatches; batch++) {
			double[][] examples = new double[batchSize][imageExample.length];
			int[][] outcomeMatrix = new int[batchSize][10];

			for(int i = 1 + batch; i < batchSize + 1 + batch; i++) {
				//1 based indices
				man.setCurrent(i);
				double[] currExample = ArrayUtil.flatten(ArrayUtil.toDouble(man.readImage()));
				examples[i - 1 - batch] = currExample;
				outcomeMatrix[i - 1 - batch] = ArrayUtil.toOutcomeArray(man.readLabel(), 10);
			}
			DoubleMatrix training = new DoubleMatrix(examples);
			ret.add(new Pair<>(training,MatrixUtil.toMatrix(outcomeMatrix)));
		}

		return ret;
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
	public static Pair<DoubleMatrix,DoubleMatrix> getMnistExampleBatch(int batchSize) throws IOException {
		File ensureExists = new File("/tmp/MNIST");
		if(!ensureExists.exists() || !new File("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped).exists() || !new File("/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped).exists()) 
			new MnistFetcher().downloadAndUntar();
		MnistManager man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);

		int[][] image = man.readImage();
		int[] imageExample = ArrayUtil.flatten(image);
		double[][] examples = new double[batchSize][imageExample.length];
		int[][] outcomeMatrix = new int[batchSize][10];
		for(int i = 1; i < batchSize + 1; i++) {
			//1 based indices
			man.setCurrent(i);
			double[] currExample = ArrayUtil.flatten(ArrayUtil.toDouble(man.readImage()));
			for(int j = 0; j < currExample.length; j++)
				currExample[j] = MathUtils.normalize(currExample[j], 0, 255);
			examples[i - 1] = currExample;
			outcomeMatrix[i - 1] = ArrayUtil.toOutcomeArray(man.readLabel(), 10);
		}
		DoubleMatrix training = new DoubleMatrix(examples);
		return new Pair<DoubleMatrix,DoubleMatrix>(training,MatrixUtil.toMatrix(outcomeMatrix));

	}


}





