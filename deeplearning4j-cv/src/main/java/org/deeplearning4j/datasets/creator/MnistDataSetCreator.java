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

package org.deeplearning4j.datasets.creator;

import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;

public class MnistDataSetCreator {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(60000);
		DataSet save = fetcher.next();
        SerializationUtils.saveObject(save,new File(args[0]));

	}

}
