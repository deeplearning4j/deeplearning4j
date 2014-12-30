package org.deeplearning4j.base;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistFetcher {

	private File fileDir;
	private static final Logger LOG = LoggerFactory.getLogger(MnistFetcher.class);
	private static final String trainingFilesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";

	private static final String trainingFilesFilename = "images-idx1-ubyte.gz";
	public static final String trainingFilesFilename_unzipped = "images-idx1-ubyte";

	private static final String trainingFileLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
	private static final String trainingFileLabelsFilename = "labels-idx1-ubyte.gz";
	public static final String trainingFileLabelsFilename_unzipped = "labels-idx1-ubyte";
	private static final String LOCAL_DIR_NAME = "MNIST";



	public  File downloadAndUntar() throws IOException {
		if(fileDir != null) {
			return fileDir;
		}
		// mac gives unique tmp each run and we want to store this persist
		// this data across restarts
		File tmpDir = new File(System.getProperty("user.home"));

		File baseDir = new File(tmpDir, LOCAL_DIR_NAME);
		if(!(baseDir.isDirectory() || baseDir.mkdir())) {
			throw new IOException("Could not mkdir " + baseDir);
		}


		LOG.info("Downloading mnist...");
		// getFromOrigin training records
		File tarFile = new File(baseDir, trainingFilesFilename);

		if(!tarFile.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile);
		}

		ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(),baseDir.getAbsolutePath());

		// getFromOrigin training records
		File labels = new File(baseDir, trainingFileLabelsFilename);

		if(!labels.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), labels);
		}

		ArchiveUtils.unzipFileTo(labels.getAbsolutePath(),baseDir.getAbsolutePath());

		fileDir = baseDir;
		return fileDir;
	}
}
