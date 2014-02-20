package org.deeplearning4j.datasets.fetchers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.base.MnistFetcher;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.MnistManager;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;


/**
 * Data fetcher for the MNIST dataset
 * @author Adam Gibson
 *
 */
public class MnistDataFetcher extends BaseDataFetcher {

	private transient MnistManager man;
	public final static int NUM_EXAMPLES = 60000;

	public MnistDataFetcher() throws IOException {
		if(!new File("/tmp/mnist").exists())
			new MnistFetcher().downloadAndUntar();
		man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
		numOutcomes = 10;
		totalExamples = NUM_EXAMPLES;
		//1 based cursor
		cursor = 1;
		man.setCurrent(cursor);
		int[][] image;
		try {
			image = man.readImage();
		} catch (IOException e) {
			throw new IllegalStateException("Unable to read image");
		}
		inputColumns = ArrayUtil.flatten(image).length;


	}

	@Override
	public void fetch(int numExamples) {
		if(!hasMore())
			throw new IllegalStateException("Unable to get more; there are no more images");



		//we need to ensure that we don't overshoot the number of examples total
		List<Pair<DoubleMatrix,DoubleMatrix>> toConvert = new ArrayList<>();

		for(int i = 0; i < numExamples; i++,cursor++) {
			if(!hasMore())
				break;
			if(man == null) {
				try {
					man = new MnistManager("/tmp/MNIST/" + MnistFetcher.trainingFilesFilename_unzipped,"/tmp/MNIST/" + MnistFetcher.trainingFileLabelsFilename_unzipped);
				} catch (IOException e) {
					throw new RuntimeException(e);
				}
			}
			man.setCurrent(cursor);
			//note data normalization
			try {
				DoubleMatrix in = MatrixUtil.toMatrix(ArrayUtil.flatten(man.readImage()));
				for(int d = 0; d < in.length; d++) {
					if(in.get(d) > 30) {
						in.put(d,1);
					}
					else 
						in.put(d,0);
				}


				DoubleMatrix out = createOutputVector(man.readLabel());
				boolean found = false;
				for(int col = 0; col < out.length; col++) {
					if(out.get(col) > 0) {
						found = true;
						break;
					}
				}
				if(!found)
					throw new IllegalStateException("Found a matrix without an outcome");

				toConvert.add(new Pair<>(in,out));
			} catch (IOException e) {
				throw new IllegalStateException("Unable to read image");

			}
		}


		initializeCurrFromList(toConvert);



	}

	@Override
	public void reset() {
		cursor = 1;
	}





}
