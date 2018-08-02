/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.base;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.common.resources.ResourceType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.resources.Downloader;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class IrisUtils {

    private static final String IRIS_RELATIVE_URL = "datasets/iris.dat";
    private static final String MD5 = "1c21400a78061197eac64c6748844216";

    private IrisUtils() {}

    public static List<DataSet> loadIris(int from, int to) throws IOException {
        File rootDir = DL4JResources.getDirectory(ResourceType.DATASET, "iris");
        File irisData = new File(rootDir, "iris.dat");
        if(!irisData.exists()){
            URL url = DL4JResources.getURL(IRIS_RELATIVE_URL);
            Downloader.download("Iris", url, irisData, MD5, 3);
        }

        @SuppressWarnings("unchecked")
        List<String> lines;
        try(InputStream is = new FileInputStream(irisData)){
            lines = IOUtils.readLines(is);
        }
        List<DataSet> list = new ArrayList<>();
        INDArray ret = Nd4j.ones(Math.abs(to - from), 4);
        double[][] outcomes = new double[lines.size()][3];
        int putCount = 0;

        for (int i = from; i < to; i++) {
            String line = lines.get(i);
            String[] split = line.split(",");

            addRow(ret, putCount++, split);

            String outcome = split[split.length - 1];
            double[] rowOutcome = new double[3];
            rowOutcome[Integer.parseInt(outcome)] = 1;
            outcomes[i] = rowOutcome;
        }

        for (int i = 0; i < ret.rows(); i++) {
            DataSet add = new DataSet(ret.getRow(i), Nd4j.create(outcomes[from + i]));
            list.add(add);
        }
        return list;
    }

    private static void addRow(INDArray ret, int row, String[] line) {
        double[] vector = new double[4];
        for (int i = 0; i < 4; i++)
            vector[i] = Double.parseDouble(line[i]);

        ret.putRow(row, Nd4j.create(vector));
    }
}
