/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.base;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.util.ArchiveUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
@Data
@NoArgsConstructor
public class MnistFetcher {
	protected static final Logger log = LoggerFactory.getLogger(MnistFetcher.class);

	protected File BASE_DIR = new File(System.getProperty("user.home"));
	protected static final String LOCAL_DIR_NAME = "MNIST";
	protected File FILE_DIR = new File(BASE_DIR, LOCAL_DIR_NAME);

	private File fileDir;
	private static final String trainingFilesURL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
	private static final String trainingFilesFilename = "images-idx3-ubyte.gz";
	public static final String trainingFilesFilename_unzipped = "images-idx3-ubyte";
	private static final String trainingFileLabelsURL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
	private static final String trainingFileLabelsFilename = "labels-idx1-ubyte.gz";
	public static final String trainingFileLabelsFilename_unzipped = "labels-idx1-ubyte";

	//Test data:
	private static final String testFilesURL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz";
	private static final String testFilesFilename = "t10k-images-idx3-ubyte.gz";
	public static final String testFilesFilename_unzipped = "t10k-images-idx3-ubyte";
	private static final String testFileLabelsURL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz";
	private static final String testFileLabelsFilename = "t10k-labels-idx1-ubyte.gz";
	public static final String testFileLabelsFilename_unzipped = "t10k-labels-idx1-ubyte";


	public  File downloadAndUntar() throws IOException {
		if (fileDir != null) {
			return fileDir;
		}

		File baseDir = FILE_DIR;
		if (!(baseDir.isDirectory() || baseDir.mkdir())) {
			throw new IOException("Could not mkdir " + baseDir);
		}

		log.info("Downloading mnist...");
		// getFromOrigin training records
		File tarFile = new File(baseDir, trainingFilesFilename);
		File tarFileLabels = new File(baseDir, testFilesFilename);

		if (!tarFile.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFilesURL), tarFile);
		}

		if (!tarFileLabels.isFile()) {
			FileUtils.copyURLToFile(new URL(testFilesURL), tarFileLabels);
		}

		ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), baseDir.getAbsolutePath());
		ArchiveUtils.unzipFileTo(tarFileLabels.getAbsolutePath(), baseDir.getAbsolutePath());

		// getFromOrigin training records
		File labels = new File(baseDir, trainingFileLabelsFilename);
		File labelsTest = new File(baseDir, testFileLabelsFilename);

		if (!labels.isFile()) {
			FileUtils.copyURLToFile(new URL(trainingFileLabelsURL), labels);
		}
		if (!labelsTest.isFile()) {
			FileUtils.copyURLToFile(new URL(testFileLabelsURL), labelsTest);
		}

		ArchiveUtils.unzipFileTo(labels.getAbsolutePath(), baseDir.getAbsolutePath());
		ArchiveUtils.unzipFileTo(labelsTest.getAbsolutePath(), baseDir.getAbsolutePath());

		fileDir = baseDir;
		return fileDir;
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
