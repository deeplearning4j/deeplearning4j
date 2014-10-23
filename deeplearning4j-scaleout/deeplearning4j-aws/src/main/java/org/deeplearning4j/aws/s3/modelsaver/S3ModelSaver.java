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
import org.deeplearning4j.iterativereduce.actor.core.ModelSaver;
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
		S3Uploader uploader = new S3Uploader();
		String tmpDir = System.getProperty("java.io.tmpdir");;
		try {
			File tmpFile = new File(new File(tmpDir),modelId);
			ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(tmpFile));
			os.writeObject(ser);
			os.flush();
			os.close();

			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(tmpFile));
			uploader.upload(ois, modelId, bucketName);
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
