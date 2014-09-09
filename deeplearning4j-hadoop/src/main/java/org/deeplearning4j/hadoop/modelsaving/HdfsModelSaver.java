package org.deeplearning4j.hadoop.modelsaving;

import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.deeplearning4j.iterativereduce.actor.core.ModelSaver;
/**
 * Saves models to hdfs
 * @author Adam Gibson
 *
 */
public class HdfsModelSaver implements ModelSaver {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3274766215282776643L;
	private Configuration conf;
	private Path path;
	private boolean reWrite;
	
	/**
	 * A config and baseline path are needed.
	 * The path is required here so that there 
	 * is more to think about in terms of permissions.
	 * @param conf the conf to use
	 * @param path the path to the file to write to
	 */
	public HdfsModelSaver(Configuration conf,Path path) {
		super();
		this.conf = conf;
		this.path = path;
	}



	@Override
	public void save(Serializable ser) {
		FileSystem system;
		try {
			system = FileSystem.get(conf);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		if(reWrite) {
			try {
				system.rename(path, new Path(path.getParent(),path.getName() + System.currentTimeMillis()));
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		else {
			try {
				OutputStream os = system.create(path);
				ObjectOutputStream bos = new ObjectOutputStream(os);
				bos.writeObject(ser);
				bos.flush();
				bos.close();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
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
