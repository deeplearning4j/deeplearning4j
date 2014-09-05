package org.deeplearning4j.datasets.creator;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.util.SerializationUtils;

public class MnistDataSetCreator {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(60000);
		DataSet save = fetcher.next();
        SerializationUtils.saveObject(save,new File(args[0]));

	}

}
