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

package org.deeplearning4j.aws.s3.modelsaver;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.UUID;

import org.deeplearning4j.aws.s3.BaseS3;
import org.deeplearning4j.aws.s3.uploader.S3Uploader;
import org.deeplearning4j.scaleout.actor.core.ModelSaver;

/**
 * Saves models to a specified bucket name
 * @author Adam Gibson
 *
 */
public class S3ModelSaver extends BaseS3 implements ModelSaver {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3922672528784966211L;

	private String bucketName;



	public S3ModelSaver(String bucketName) {
		super();
		this.bucketName = bucketName;
	}



	@Override
	public void save(Serializable ser) {
		String modelId = UUID.randomUUID().toString();
		save(ser, modelId);
	}

	public void save(Serializable ser, String name) {
		S3Uploader uploader = new S3Uploader();

		try {
			File tmpFile = File.createTempFile("temp","upload");
			tmpFile.deleteOnExit();
			ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(tmpFile));
			os.writeObject(ser);
			os.flush();
			os.close();

			//ObjectInputStream ois = new ObjectInputStream(new FileInputStream(tmpFile));
			//uploader.upload(ois, name, bucketName, tmpFile.length());
			uploader.upload(tmpFile, name, bucketName);
		}catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

    /**
     * Returns whether a model exists or not
     *
     * @return true if the model exists, false otherwise
     */
    @Override
    public boolean exists() {
        throw new UnsupportedOperationException();
    }

    /**
     * Loads the model from the location that it saves to
     *
     * @param type
     */
    @Override
    public <E> E load(Class<E> type) {
        throw new UnsupportedOperationException();
    }
}
