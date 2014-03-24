package org.deeplearning4j.iterativereduce.actor.core;

import java.io.File;
import java.io.Serializable;

import org.deeplearning4j.util.SerializationUtils;
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


		SerializationUtils.saveObject(ser, file);
	}


}
