package org.deeplearning4j.aws.s3.reader;

import com.amazonaws.services.s3.AmazonS3;

/**
 * When paginating through a result applyTransformToDestination,
 * allows the user to react to a bucket result being found
 * @author Adam Gibson
 *
 */
public interface BucketKeyListener {

	/**
	 * 
	 * @param s3 an s3 client
	 * @param bucket the bucket being iterated on
	 * @param key the current key
	 */
	void onKey(AmazonS3 s3,String bucket,String key);
	
	
}
