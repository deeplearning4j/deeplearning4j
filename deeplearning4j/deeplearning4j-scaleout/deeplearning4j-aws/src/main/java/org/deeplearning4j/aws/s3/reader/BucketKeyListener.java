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
