package org.deeplearning4j.util;

import java.io.*;
import java.util.logging.Logger;

public class FileOperations {

	private static final Logger LOG = Logger.getLogger(FileOperations.class.getName());

	private FileOperations() {}

	public static OutputStream createAppendingOutputStream(File to) {
		try {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to,true));
			return bos;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static void appendTo(String data,File append) {
		try {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(append,true));
			bos.write(data.getBytes());
			bos.flush();
			bos.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
	}

	public  void untarFile(File baseDir, File tarFile) throws IOException {

		LOG.info("Untaring File: " + tarFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("tar -C %s -xvf %s",
				baseDir.getAbsolutePath(), tarFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new
				InputStreamReader(p.getErrorStream()));
		LOG.info("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			LOG.info(s);
		}
		stdError.close();


	}

	public static void gunzipFile(File baseDir, File gzFile) throws IOException {

		LOG.info("gunzip'ing File: " + gzFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("gunzip %s",
				gzFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new
				InputStreamReader(p.getErrorStream()));
		LOG.info("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			LOG.info(s);
		}
		stdError.close();
	}


}
