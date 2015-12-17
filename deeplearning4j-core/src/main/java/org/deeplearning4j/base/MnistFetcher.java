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

	public static Map<String, String> mnistTrainData = new HashMap<>();
	public static Map<String, String> mnistTrainLabel = new HashMap<>();
	public static Map<String, String> mnistTestData = new HashMap<>();
	public static Map<String, String> mnistTestLabel = new HashMap<>();

	public void generateMnistFileMaps(){
		mnistTrainData.put("filesFilename", "images-idx1-ubyte.gz");
		mnistTrainData.put("filesURL","http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
		mnistTrainData.put("filesFilenameUnzipped","images-idx1-ubyte");

		mnistTrainLabel.put("fileFilename", "labels-idx1-ubyte.gz");
		mnistTrainLabel.put("fileURL", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
		mnistTrainLabel.put("fileFilenameUnzipped","labels-idx1-ubyte");

		mnistTestData.put("filesFilename", "t10k-images-idx3-ubyte.gz");
		mnistTestData.put("filesURL","http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz");
		mnistTestData.put("filesFilenameUnzipped","t10k-images-idx3-ubyte");

		mnistTestLabel.put("fileFilename", "t10k-labels-idx1-ubyte.gz");
		mnistTestLabel.put("fileURL", "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz");
		mnistTestLabel.put("fileFilenameUnzipped","t10k-labels-idx1-ubyte");

	}

	public void downloadAndUntar(Map urlMap) throws IOException {
		File tarFile = new File(FILE_DIR, urlMap.get("filesFilename").toString());
		if (!tarFile.isFile()) {
			FileUtils.copyURLToFile(new URL(urlMap.get("filesURL").toString()), tarFile);
		}
		ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), FILE_DIR.getAbsolutePath());

	}
	
	public void fetchMnist() throws IOException {
		if(!FILE_DIR.exists()) {

			// mac gives unique tmp each run and we want to store this persist
			// this data across restarts

			if (!(FILE_DIR.isDirectory() || FILE_DIR.mkdir())) {
				throw new IOException("Could not mkdir " + FILE_DIR);
			}

			generateMnistFileMaps();

			log.info("Downloading mnist...");
			downloadAndUntar(mnistTrainData);
			downloadAndUntar(mnistTrainLabel);
			downloadAndUntar(mnistTestData);
			downloadAndUntar(mnistTestLabel);

		}
	}


}
