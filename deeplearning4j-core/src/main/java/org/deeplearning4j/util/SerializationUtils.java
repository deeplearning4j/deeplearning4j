package org.deeplearning4j.util;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.apache.commons.io.FileUtils;

public class SerializationUtils {

	@SuppressWarnings("unchecked")
	public static <T> T readObject(File file) {
		try {
			ObjectInputStream ois = new ObjectInputStream(FileUtils.openInputStream(file));
			return (T) ois.readObject();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
	}
	
	/**
	 * Reads an object from the given input stream
	 * @param is the input stream to read from
	 * @return the read object
	 */
	@SuppressWarnings("unchecked")
	public static <T> T readObject(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			return (T) ois.readObject();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
	}
	
	/**
	 * Writes the object to the output stream
	 * THIS DOES NOT FLUSH THE STREAM
	 * @param toSave the object to save
	 * @param writeTo the output stream to write to
	 */
	public static void writeObject(Serializable toSave,OutputStream writeTo) {
		try {
			ObjectOutputStream os = new ObjectOutputStream(writeTo);
			os.writeObject(toSave);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
	}
	
	public static void saveObject(Object toSave,File saveTo) {
		try {
			ObjectOutputStream os = new ObjectOutputStream(FileUtils.openOutputStream(saveTo));
			os.writeObject(toSave);
			os.flush();
			os.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		
	}
}
