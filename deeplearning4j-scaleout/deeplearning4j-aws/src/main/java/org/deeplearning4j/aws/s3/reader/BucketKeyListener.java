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
    void onKey(AmazonS3 s3, String bucket, String key);


}
