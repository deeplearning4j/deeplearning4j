package org.deeplearning4j.aws.s3.reader;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.aws.s3.BaseS3;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.S3Object;
/**
 * Downloads files from S3
 * @author Adam Gibson
 *
 */
public class S3Downloader extends BaseS3 {

	
	
	
	public void download(String bucket,String key,File to) throws IOException {
		AmazonS3 s3 = getClient();
		S3Object obj = s3.getObject(bucket, key);
		InputStream is = obj.getObjectContent();
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to));
		IOUtils.copy(is, bos);
		bos.close();
		is.close();
		obj.close();
	}
	
	public void download(String bucket,String key,OutputStream to) throws IOException {
		AmazonS3 s3 = getClient();
		S3Object obj = s3.getObject(bucket, key);
		InputStream is = obj.getObjectContent();
		BufferedOutputStream bos = new BufferedOutputStream(to);
		
		IOUtils.copy(is, bos);
		bos.close();
		is.close();
		obj.close();
	}
	
}
