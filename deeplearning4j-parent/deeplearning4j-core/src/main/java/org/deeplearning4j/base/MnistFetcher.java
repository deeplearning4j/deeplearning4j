package org.deeplearning4j.base;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;

import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistFetcher {

	private File fileDir;
	private static Logger log = LoggerFactory.getLogger(MnistFetcher.class);
	private static final String trainingFilesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";

	private static final String trainingFilesFilename = "train-images-idx3-ubyte.gz";
	public static final String trainingFilesFilename_unzipped = "train-images-idx3-ubyte";

	private static final String trainingFileLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
	private static final String trainingFileLabelsFilename = "train-labels-idx1-ubyte.gz";
	public static final String trainingFileLabelsFilename_unzipped = "train-labels-idx1-ubyte";
	private static final String LOCAL_DIR_NAME = "MNIST";

	
	
	public  File downloadAndUntar() throws IOException {
		if(fileDir != null) {
			return fileDir;
		}
		// mac gives unique tmp each run and we want to store this persist
		// this data across restarts
		File tmpDir = new File("/tmp");
		if(!tmpDir.isDirectory()) {
			tmpDir = new File(System.getProperty("java.io.tmpdir"));
		}
		File baseDir = new File(tmpDir, LOCAL_DIR_NAME);
		if(!(baseDir.isDirectory() || baseDir.mkdir())) {
			throw new IOException("Could not mkdir " + baseDir);
		}



		// get training records
		File tarFile = new File(baseDir, trainingFilesFilename);

		if(!tarFile.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile);      
		}

		gunzipFile(baseDir, tarFile);

		// get training records labels - trainingFileLabelsURL
		File tarLabelsFile = new File(baseDir, trainingFileLabelsFilename);

		if(!tarLabelsFile.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), tarLabelsFile);      
		}

		gunzipFile(baseDir, tarLabelsFile);





		fileDir = baseDir;
		return fileDir;
	}

	public  void untarFile(File baseDir, File tarFile) throws IOException {

		log.info("Untaring File: " + tarFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("tar -C %s -xvf %s", 
				baseDir.getAbsolutePath(), tarFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new 
				InputStreamReader(p.getErrorStream()));
		log.info("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			log.info(s);
		}
		stdError.close();


	}

	public static void gunzipFile(File baseDir, File gzFile) throws IOException {

		log.info("gunzip'ing File: " + gzFile.toString());

		Process p = Runtime.getRuntime().exec(String.format("gunzip %s", 
				gzFile.getAbsolutePath()));
		BufferedReader stdError = new BufferedReader(new 
				InputStreamReader(p.getErrorStream()));
		log.info("Here is the standard error of the command (if any):\n");
		String s;
		while ((s = stdError.readLine()) != null) {
			log.info(s);
		}
		stdError.close();


	}


}
