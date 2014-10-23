package org.deeplearning4j.aws.s3.reader;

import java.io.InputStream;
import java.util.Iterator;
import java.util.List;

import com.amazonaws.services.s3.model.ObjectListing;
import com.amazonaws.services.s3.model.S3ObjectSummary;
/**
 * Iterator over individual S3 bucket
 * @author Adam Gibson
 *
 */
public class BucketIterator implements Iterator<InputStream> {

	private S3Downloader s3;
	private String bucket;
	private ObjectListing currList;
	private List<S3ObjectSummary> currObjects;
	private int currObject;

	
	
	
	public BucketIterator(String bucket) {
		this(bucket,null);
		
	}
	
	
	public BucketIterator(String bucket,S3Downloader s3) {
		this.bucket = bucket;
		
		if(s3 == null)
			s3 = new S3Downloader();
		currList = s3.listObjects(bucket);
		currObjects = currList.getObjectSummaries();
		
	}
	
	

	@Override
	public boolean hasNext() {
	    return !currList.isTruncated() && currObject < currObjects.size(); 
	}

	@Override
	public InputStream next() {
		if(currObject < currObjects.size()) {
			InputStream ret = s3.objectForKey(bucket, currObjects.get(currObject).getKey());
			currObject++;
			return ret;
		}
		else if(currList.isTruncated()){
			currList = s3.nextList(currList);
			currObjects = currList.getObjectSummaries();
			currObject = 0;
			
			InputStream ret = s3.objectForKey(bucket, currObjects.get(currObject).getKey());
			
			currObject++;
			return ret;
		}
		
		
		return null;
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}


}
