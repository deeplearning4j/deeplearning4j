/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.aws.s3.reader;

import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.model.*;
import com.amazonaws.services.s3.transfer.MultipleFileDownload;
import com.amazonaws.services.s3.transfer.TransferManager;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.aws.s3.BaseS3;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Downloads files from S3
 * @author Adam Gibson
 *
 */
public class S3Downloader extends BaseS3 {


    /**
     * Return the keys for a bucket
     * @param bucket the bucket to get the keys for
     * @return the bucket's keys
     */
    public List<String> keysForBucket(String bucket) {
        AmazonS3 s3 = getClient();
        List<String> ret = new ArrayList<>();
        ListObjectsRequest listObjectsRequest = new ListObjectsRequest().withBucketName(bucket);
        ObjectListing objectListing;

        do {
            objectListing = s3.listObjects(listObjectsRequest);
            for (S3ObjectSummary objectSummary : objectListing.getObjectSummaries()) {
                ret.add(objectSummary.getKey());
            }
            listObjectsRequest.setMarker(objectListing.getNextMarker());
        } while (objectListing.isTruncated());

        return ret;
    }

    /**
     * Returns the list of buckets in s3
     * @return the list of buckets
     */
    public List<String> buckets() {
        List<String> ret = new ArrayList<>();
        AmazonS3 s3 = getClient();
        List<Bucket> buckets = s3.listBuckets();
        for (Bucket b : buckets)
            ret.add(b.getName());
        return ret;
    }

    /**
     * Iterate over individual buckets.
     * Returns input streams to each object.
     * It is your responsibility to close the input streams
     * @param bucket the bucket to iterate over
     * @return an iterator over the objects in an s3 bucket
     */
    public Iterator<InputStream> iterateBucket(String bucket) {
        return new BucketIterator(bucket, this);
    }

    /**
     * Iterator style one list at a time
     * @param list the list to get the next batch for
     * @return the next batch of objects or null if
     * none are left
     */
    public ObjectListing nextList(ObjectListing list) {
        AmazonS3 s3 = getClient();
        if (list.isTruncated())
            return s3.listNextBatchOfObjects(list);
        return null;
    }

    /**
     * Simple way of retrieving the listings for a bucket
     * @param bucket the bucket to retrieve listings for
     * @return the object listing for this bucket
     */
    public ObjectListing listObjects(String bucket) {
        AmazonS3 s3 = getClient();
        ObjectListing list = s3.listObjects(bucket);
        return list;
    }

    /**
     * Paginates through a bucket's keys invoking the listener 
     * at each key
     * @param bucket the bucket to iterate
     * @param listener the listener
     */
    public void paginate(String bucket, BucketKeyListener listener) {
        AmazonS3 s3 = getClient();
        ObjectListing list = s3.listObjects(bucket);
        for (S3ObjectSummary summary : list.getObjectSummaries()) {
            if (listener != null)
                listener.onKey(s3, bucket, summary.getKey());
        }

        while (list.isTruncated()) {
            list = s3.listNextBatchOfObjects(list);
            for (S3ObjectSummary summary : list.getObjectSummaries()) {
                if (listener != null)
                    listener.onKey(s3, bucket, summary.getKey());
            }
        }


    }


    /**
     * Returns an input stream for the given bucket and key
     * @param bucket the bucket to retrieve from
     * @param key the key of the objec  t
     * @return an input stream to the object
     */
    public InputStream objectForKey(String bucket, String key) {
        AmazonS3 s3 = getClient();
        S3Object obj = s3.getObject(bucket, key);
        InputStream is = obj.getObjectContent();
        return is;
    }


    public void download(String bucket, String key, File to) throws IOException {
        AmazonS3 s3 = getClient();
        S3Object obj = s3.getObject(bucket, key);
        InputStream is = obj.getObjectContent();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to));
        IOUtils.copy(is, bos);
        bos.close();
        is.close();
        obj.close();
    }

    public void download(String bucket, String key, OutputStream to) throws IOException {
        AmazonS3 s3 = getClient();
        S3Object obj = s3.getObject(bucket, key);
        InputStream is = obj.getObjectContent();
        BufferedOutputStream bos = new BufferedOutputStream(to);

        IOUtils.copy(is, bos);
        bos.close();
        is.close();
        obj.close();
    }

    public MultipleFileDownload downloadFolder(String bucketName, String keyPrefix, File folderPath) {
        TransferManager transfer = new TransferManager(getClient());
        return transfer.downloadDirectory(bucketName, keyPrefix, folderPath);
    }


}
