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

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.InputStream;
import java.util.Iterator;
import java.util.List;

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
        if (currIterator.hasNext())
            return currIterator.next();
        else if (currBucket < buckets.size()) {
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
