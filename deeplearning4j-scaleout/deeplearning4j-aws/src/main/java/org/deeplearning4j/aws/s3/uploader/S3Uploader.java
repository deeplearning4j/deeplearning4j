package org.deeplearning4j.aws.s3.uploader;

import java.io.File;
import java.io.InputStream;
import java.util.List;

import org.deeplearning4j.aws.s3.BaseS3;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3Client;
import com.amazonaws.services.s3.model.Bucket;
import com.amazonaws.services.s3.model.ObjectMetadata;

/**
 * Uploads files to S3
 * 
 * @see {@link BaseS3}
 * @author Adam Gibson
 *
 */
public class S3Uploader extends BaseS3 {


	
	
	public void upload(File file,String bucketName) {
		AmazonS3 client = new AmazonS3Client(creds);
		List<Bucket> buckets = client.listBuckets();
		for(Bucket b : buckets) 
			if(b.getName().equals(bucketName)) {
				client.putObject(bucketName, file.getName(), file);
				return;
			}
		
		//bucket didn't exist: create it
		client.createBucket(bucketName);
		client.putObject(bucketName, file.getName(), file);

		
	}

	public void upload(InputStream is,String name,String bucketName) {
		AmazonS3 client = getClient();
	
		List<Bucket> buckets = client.listBuckets();
		ObjectMetadata med = new ObjectMetadata();
		for(Bucket b : buckets) 
			if(b.getName().equals(bucketName)) {
				client.putObject(bucketName, name, is, med);
				return;
			}
		
		//bucket didn't exist: create it
		client.createBucket(bucketName);
		client.putObject(bucketName, name, is, med);

		
	}
	
	
	

	

	
	
	
	

}
