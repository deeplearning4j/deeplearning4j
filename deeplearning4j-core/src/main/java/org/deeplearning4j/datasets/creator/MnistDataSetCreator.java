package org.deeplearning4j.datasets.creator;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;

public class MnistDataSetCreator {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(60000);
		DataSet save = fetcher.next();
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-data.bin"));
		save.write(bos);
		bos.flush();
		bos.close();
	}

}
