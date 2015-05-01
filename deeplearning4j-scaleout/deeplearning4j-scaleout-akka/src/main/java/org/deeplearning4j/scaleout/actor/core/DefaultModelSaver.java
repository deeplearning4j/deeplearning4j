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

package org.deeplearning4j.scaleout.actor.core;

import java.io.File;
import java.io.Serializable;

import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Saves a file to the relative path nn-model.bin
 * or another location specified by the user.
 * Renames the file to another file indicated by the timestamp
 * if this file already exists.
 * @author Adam Gibson
 *
 */
public class DefaultModelSaver implements ModelSaver {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7563476748417989927L;
	private File file;
	private boolean rewrite = false;
    private static final Logger log = LoggerFactory.getLogger(DefaultModelSaver.class);

	public DefaultModelSaver(File file) {
		this(file,false);
	}



	public DefaultModelSaver(File file, boolean rewrite) {
		super();
		this.file = file;
		this.rewrite = rewrite;
	}



	public DefaultModelSaver() {
		super();
		file = new File("nn-model.bin");
	}



	@Override
	public void save(Serializable ser)  {
		if(!rewrite) {
			String path = file.getAbsolutePath();
			if(file.exists()) {
				file.renameTo(new File(file.getParentFile(),file.getName() + "-" + System.currentTimeMillis()));
			}

			file = new File(path);

		}
        log.info("Saving to " + file);

		SerializationUtils.saveObject(ser, file);


	}

    /**
     * Returns whether a model exists or not
     *
     * @return true if the model exists, false otherwise
     */
    @Override
    public boolean exists() {
       return file.exists();
    }

    /**
     * Loads the model from the location that it saves to
     */
    @Override
    public <E>  E load(Class<E> clazz) {
        return SerializationUtils.readObject(file);
    }
}
