/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.aws.s3.uploader;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.amazonaws.services.s3.model.*;
import com.amazonaws.services.s3.transfer.MultipleFileUpload;
import com.amazonaws.services.s3.transfer.TransferManager;
import com.amazonaws.services.s3.transfer.Upload;
import org.deeplearning4j.aws.s3.BaseS3;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3Client;

/**
 * Uploads files to S3
 * 
 * @see {@link BaseS3}
 * @author Adam Gibson
 *
 */
public class S3Uploader extends BaseS3 {


    /**
     * Multi part upload for big files
     * @param file the file to upload
     * @param bucketName the bucket name to upload
     */
    public void multiPartUpload(File file,String bucketName) {
        AmazonS3 client = new AmazonS3Client(creds);
        bucketName = ensureValidBucketName(bucketName);

        List<Bucket> buckets = client.listBuckets();
        for(Bucket b : buckets)
            if(b.getName().equals(bucketName)) {
                doMultiPart(client,bucketName,file);
                return;
            }

        //bucket didn't exist: create it
        client.createBucket(bucketName);
        doMultiPart(client,bucketName, file);
    }

    /**
     * Upload the file to the bucket.
     * Will create the bucket if it hasn't already been created
     * @param file the file to upload
     * @param bucketName the name of the bucket
     */
	public void upload(File file,String bucketName) {
		AmazonS3 client = new AmazonS3Client(creds);
		bucketName = ensureValidBucketName(bucketName);

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

    private void doMultiPart(AmazonS3 s3Client,String bucketName,File file) {
        // Create a list of UploadPartResponse objects. You get one of these
        // for each part upload.
        List<PartETag> partETags = new ArrayList<>();

        // Step 1: Initialize.
        InitiateMultipartUploadRequest initRequest = new
                InitiateMultipartUploadRequest(bucketName, file.getName());
        InitiateMultipartUploadResult initResponse =
                s3Client.initiateMultipartUpload(initRequest);

        long contentLength = file.length();
        long partSize = 5242880; // Set part size to 5 MB.

        try {
            // Step 2: Upload parts.
            long filePosition = 0;
            for (int i = 1; filePosition < contentLength; i++) {
                // Last part can be less than 5 MB. Adjust part size.
                partSize = Math.min(partSize, (contentLength - filePosition));

                // Create request to upload a part.
                UploadPartRequest uploadRequest = new UploadPartRequest()
                        .withBucketName(bucketName).withKey(file.getName())
                        .withUploadId(initResponse.getUploadId()).withPartNumber(i)
                        .withFileOffset(filePosition)
                        .withFile(file)
                        .withPartSize(partSize);

                // Upload part and add response to our list.
                partETags.add(
                        s3Client.uploadPart(uploadRequest).getPartETag());

                filePosition += partSize;
            }

            // Step 3: Complete.
            CompleteMultipartUploadRequest compRequest = new
                    CompleteMultipartUploadRequest(
                    bucketName,
                    file.getName(),
                    initResponse.getUploadId(),
                    partETags);

            s3Client.completeMultipartUpload(compRequest);
        } catch (Exception e) {
            s3Client.abortMultipartUpload(new AbortMultipartUploadRequest(
                    bucketName, file.getName(), initResponse.getUploadId()));
        }
    }

	private String ensureValidBucketName(String bucketName) {
		String formatted = bucketName.replaceAll("\\s+","_");
		int length = bucketName.length();
		if(length >= 62)
			length = 62;
		formatted = formatted.substring(0,length);
		formatted = formatted.replace(".","d");
		formatted = formatted.toLowerCase();
		if(formatted.endsWith("-"))
			formatted = formatted.substring(0,length - 1);
		
		return formatted;
	}
	
	public void upload(File file,String name,String bucketName) {
		AmazonS3 client = getClient();
		bucketName = ensureValidBucketName(bucketName);
		List<Bucket> buckets = client.listBuckets();
	//	ObjectMetadata med = new ObjectMetadata();
//		med.setContentLength(fileLength);
		for(Bucket b : buckets)
			if(b.getName().equals(bucketName)) {
				//client.putObject(bucketName, name, is, med);
				client.putObject(new PutObjectRequest(bucketName, name, file));
				return;
			}
		
		//bucket didn't exist: createComplex it
		client.createBucket(bucketName);
		//client.putObject(bucketName, name, is, med);
		client.putObject(new PutObjectRequest(bucketName, name, file));
	}
	

	public MultipleFileUpload uploadFolder(String bucketName, String keyPrefix,  File folderPath, boolean includeSubDir) {
		TransferManager transfer = new TransferManager(getClient());
		return transfer.uploadDirectory(bucketName, keyPrefix, folderPath, includeSubDir);
	}

	public MultipleFileUpload uploadFileList(String bucketName, File folderPath, List<File> fileList, String keyPrefix){
		TransferManager transfer = new TransferManager(getClient());
		return transfer.uploadFileList(bucketName, keyPrefix, folderPath, fileList);
	}


}
