/*-
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

package org.deeplearning4j.datasets;

import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;

public class DataSets {

    private DataSets() {}

    public static DataSet mnist() {
        return mnist(60000);
    }

    public static DataSet mnist(int num) {
        try {
            MnistDataFetcher fetcher = new MnistDataFetcher();
            fetcher.fetch(num);
            return fetcher.next();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    public static DataSet iris() {
        return iris(150);
    }

    public static DataSet iris(int num) {
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(num);
        return fetcher.next();
    }

}
