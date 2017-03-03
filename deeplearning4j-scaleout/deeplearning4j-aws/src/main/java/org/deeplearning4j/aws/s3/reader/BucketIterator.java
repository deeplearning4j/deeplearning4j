/*-
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

package org.deeplearning4j.aws.s3.reader;

import com.amazonaws.services.s3.model.ObjectListing;
import com.amazonaws.services.s3.model.S3ObjectSummary;

import java.io.InputStream;
import java.util.Iterator;
import java.util.List;

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
        this(bucket, null);

    }


    public BucketIterator(String bucket, S3Downloader s3) {
        this.bucket = bucket;

        if (s3 == null)
            this.s3 = new S3Downloader();
        else
            this.s3 = s3;
        currList = this.s3.listObjects(bucket);
        currObjects = currList.getObjectSummaries();

    }



    @Override
    public boolean hasNext() {
        return currObject < currObjects.size();
    }

    @Override
    public InputStream next() {
        if (currObject < currObjects.size()) {
            InputStream ret = s3.objectForKey(bucket, currObjects.get(currObject).getKey());
            currObject++;
            return ret;
        } else if (currList.isTruncated()) {
            currList = s3.nextList(currList);
            currObjects = currList.getObjectSummaries();
            currObject = 0;

            InputStream ret = s3.objectForKey(bucket, currObjects.get(currObject).getKey());

            currObject++;
            return ret;
        }


        throw new IllegalStateException("Indeterminate state");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }


}
