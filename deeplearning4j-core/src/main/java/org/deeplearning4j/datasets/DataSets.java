package org.deeplearning4j.datasets;

import java.io.IOException;

import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.LFWDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.linalg.dataset.DataSet;

public class DataSets {

	public static DataSet mnist() {
		return mnist(60000);
	}
	
	public static DataSet mnist(int num) {
		try {
			MnistDataFetcher fetcher = new MnistDataFetcher();
			fetcher.fetch(num);
			return fetcher.next();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	

	public static DataSet lfw() {
		return lfw(LFWDataFetcher.NUM_IMAGES);
	}
	
	
	public static DataSet lfw(int num) {
		LFWDataFetcher fetcher = new LFWDataFetcher();
		fetcher.fetch(num);
		return fetcher.next();
	}
	
	public static DataSet iris() {
		return iris(150);
	}

	public static DataSet iris(int num) {
		IrisDataFetcher fetcher = new IrisDataFetcher();
		fetcher.fetch(num);
		return fetcher.next();
	}

}
