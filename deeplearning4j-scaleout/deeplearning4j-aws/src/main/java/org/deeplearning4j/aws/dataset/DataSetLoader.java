package org.deeplearning4j.aws.dataset;

import java.io.InputStream;

import org.deeplearning4j.aws.s3.reader.S3Downloader;

public class DataSetLoader {

	private String bucket;
	
	
	
	public void onData(InputStream is) {
		S3Downloader downloader = new S3Downloader();
		
	}

}
