package org.deeplearning4j.iterativereduce.runtime.io;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;


public class SVMLightHDFSDataSetIterator extends BaseDatasetIterator {

	public SVMLightHDFSDataSetIterator( int batch,int numExamples, TextRecordParser hdfsLineParser, int maxFeatureCount, int numberClasses ) throws IOException {
		
		super( batch, numExamples, new SVMLightDataFetcher( hdfsLineParser, maxFeatureCount, numberClasses ) );
		
		
	}
	
	@Override
	public boolean hasNext() {
		return fetcher.hasMore();
	}
}
