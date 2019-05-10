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

package org.deeplearning4j.aws.s3;

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.auth.PropertiesCredentials;
import com.amazonaws.services.ec2.AmazonEC2;
import com.amazonaws.services.ec2.AmazonEC2Client;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3Client;

import java.io.File;
import java.io.InputStream;


/**
 * The S3 Credentials works via discovering the credentials
 * from the system properties (passed in via -D or System wide)
 * If you invoke the JVM with -Dorg.deeplearning4j.aws.accessKey=YOUR_ACCESS_KEY
 * and -Dorg.deeplearning4j.aws.accessSecret=YOUR_SECRET_KEY
 * this will pick up the credentials from there, otherwise it will also attempt to look in
 * the system environment for the following variables:
 * 
 * 
 * AWS_ACCESS_KEY_ID
 * AWS_SECRET_ACCESS_KEY
 * @author Adam Gibson
 *
 */
public abstract class BaseS3 {


    /**
     * 
     */
    protected static final long serialVersionUID = -2280107690193651289L;
    protected String accessKey;
    protected String secretKey;
    protected AWSCredentials creds;
    public final static String ACCESS_KEY = "org.deeplearning4j.aws.accessKey";
    public final static String ACCESS_SECRET = "org.deeplearning4j.aws.accessSecret";
    public final static String AWS_ACCESS_KEY = "AWS_ACCESS_KEY"; //"AWS_ACCESS_KEY_ID";
    public final static String AWS_SECRET_KEY = "AWS_SECRET_KEY"; //"AWS_SECRET_ACCESS_KEY";


    protected void findCreds() {
        if (System.getProperty(ACCESS_KEY) != null && System.getProperty(ACCESS_SECRET) != null) {
            accessKey = System.getProperty(ACCESS_KEY);
            secretKey = System.getProperty(ACCESS_SECRET);
        }

        else if (System.getenv(AWS_ACCESS_KEY) != null && System.getenv(AWS_SECRET_KEY) != null) {
            accessKey = System.getenv(AWS_ACCESS_KEY);
            secretKey = System.getenv(AWS_SECRET_KEY);
        }
    }

    public BaseS3() {
        findCreds();
        if (accessKey != null && secretKey != null)
            creds = new BasicAWSCredentials(accessKey, secretKey);
        if (creds == null)
            throw new IllegalStateException("Unable to find ec2 credentials");
    }

    public BaseS3(File file) throws Exception {
        if (accessKey != null && secretKey != null)
            creds = new BasicAWSCredentials(accessKey, secretKey);
        else
            creds = new PropertiesCredentials(file);


    }

    public BaseS3(InputStream is) throws Exception {
        if (accessKey != null && secretKey != null)
            creds = new BasicAWSCredentials(accessKey, secretKey);
        else
            creds = new PropertiesCredentials(is);


    }

    public AWSCredentials getCreds() {
        return creds;
    }

    public void setCreds(AWSCredentials creds) {
        this.creds = creds;
    }

    public AmazonS3 getClient() {
        return new AmazonS3Client(creds);
    }

    public AmazonEC2 getEc2() {

        return new AmazonEC2Client(creds);
    }


}
