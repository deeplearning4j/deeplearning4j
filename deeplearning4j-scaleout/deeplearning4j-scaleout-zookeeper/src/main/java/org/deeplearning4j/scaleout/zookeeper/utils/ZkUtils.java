package org.deeplearning4j.scaleout.zookeeper.utils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

public class ZkUtils {

	public static byte[] toBytes(Serializable ser) throws Exception {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream ois = new ObjectOutputStream(bos);
		ois.writeObject(ser);
		return bos.toByteArray();
	}
	
	@SuppressWarnings("unchecked")
	public static <T> T fromBytes(byte[] data,Class<T> clazz) throws Exception {
		ByteArrayInputStream bis = new ByteArrayInputStream(data);
		ObjectInputStream ois = new ObjectInputStream(bis);
		return (T) ois.readObject();
	}

}
