package com.ccc.deeplearning.util;

import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

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
