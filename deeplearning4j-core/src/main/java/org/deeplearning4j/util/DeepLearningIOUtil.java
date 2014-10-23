package org.deeplearning4j.util;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

public class DeepLearningIOUtil {

	public static InputStream inputStreamFromPath(String path)  {
		try {
			return new BufferedInputStream(new FileInputStream(new File(path)));
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
}
