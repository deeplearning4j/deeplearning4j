package org.deeplearning4j.aws.s3.reader;

import java.io.InputStream;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
/**
 * baseline data applyTransformToDestination iterator for
 * @author Adam Gibson
 *
 */
public abstract class BaseS3DataSetIterator implements DataSetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 885205002006822431L;
	private S3Downloader downloader;
	private List<String> buckets;
	private int currBucket;
	private Iterator<InputStream> currIterator;
	
	public BaseS3DataSetIterator() {
		downloader = new S3Downloader();
		buckets = downloader.buckets();
		currBucket = 0;
		currIterator = downloader.iterateBucket(buckets.get(currBucket));
	}




	public InputStream nextObject() {
		if(currIterator.hasNext())
			return currIterator.next();
		else if(currBucket < buckets.size()) {
			currBucket++;
			currIterator = downloader.iterateBucket(buckets.get(currBucket));
			return currIterator.next();
		}
		
		return null;
	}



	@Override
	public boolean hasNext() {
		return currBucket < buckets.size() && currIterator.hasNext();
	}






	public String currBucket() {
		return buckets.get(currBucket);
	}



	public void nextBucket() {
		currBucket++;
	}




}
