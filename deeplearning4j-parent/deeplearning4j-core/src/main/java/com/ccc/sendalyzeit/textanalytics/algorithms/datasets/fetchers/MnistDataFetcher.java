package com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.MnistManager;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.MnistFetcher;
import com.ccc.sendalyzeit.textanalytics.util.ArrayUtil;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;

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
			man.setCurrent(cursor);
			//note data normalization
			try {
				DoubleMatrix in = MatrixUtil.toMatrix(ArrayUtil.flatten(man.readImage())).div(255);
				toConvert.add(new Pair<>(in,createOutputVector(man.readLabel())));
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
