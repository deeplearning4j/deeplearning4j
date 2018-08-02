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

package org.deeplearning4j.spark.data;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.primitives.Pair;

import java.net.URI;
import java.util.*;

/**
 * Function used with {@code RDD<MultiDataSet>.mapPartitionsWithIndex}.
 * It does two things:
 * 1. Batch MultiDataSets together, to the specified minibatch size. This may result in splitting or combining existing
 * MultiDataSet objects as required
 * 2. Export the MultiDataSet objects to the specified directory.
 * <p>
 * Naming convention for exported files:
 * "mds_" + partitionIdx + JVM_UID + "_" + idx + ".bin"
 * where 'idx' is the index of the MultiDataSet objects in this partition
 *
 * @author Alex Black
 */
public class BatchAndExportMultiDataSetsFunction
                implements Function2<Integer, Iterator<MultiDataSet>, Iterator<String>> {

    private static final Configuration conf = new Configuration();
    private final int minibatchSize;
    private final String exportBaseDirectory;
    private final String jvmuid;

    /**
     * @param minibatchSize       Minibatch size to combine examples to (if necessary)
     * @param exportBaseDirectory Base directory for exporting
     */
    public BatchAndExportMultiDataSetsFunction(int minibatchSize, String exportBaseDirectory) {
        this.minibatchSize = minibatchSize;
        this.exportBaseDirectory = exportBaseDirectory;
        String fullUID = UIDProvider.getJVMUID();
        this.jvmuid = (fullUID.length() <= 8 ? fullUID : fullUID.substring(0, 8));
    }

    @Override
    public Iterator<String> call(Integer partitionIdx, Iterator<MultiDataSet> iterator) throws Exception {

        List<String> outputPaths = new ArrayList<>();
        LinkedList<MultiDataSet> tempList = new LinkedList<>();

        int count = 0;
        while (iterator.hasNext()) {
            MultiDataSet next = iterator.next();
            if (next.getFeatures(0).size(0) == minibatchSize) {
                outputPaths.add(export(next, partitionIdx, count++));
                continue;
            }
            //DataSet must be either smaller or larger than minibatch size...
            tempList.add(next);
            Pair<Integer, List<String>> countAndPaths = processList(tempList, partitionIdx, count, false);
            if (countAndPaths.getSecond() != null && !countAndPaths.getSecond().isEmpty()) {
                outputPaths.addAll(countAndPaths.getSecond());
            }
            count = countAndPaths.getFirst();
        }

        //We might have some left-over examples...
        Pair<Integer, List<String>> countAndPaths = processList(tempList, partitionIdx, count, true);
        if (countAndPaths.getSecond() != null && !countAndPaths.getSecond().isEmpty()) {
            outputPaths.addAll(countAndPaths.getSecond());
        }

        return outputPaths.iterator();
    }

    private Pair<Integer, List<String>> processList(LinkedList<MultiDataSet> tempList, int partitionIdx,
                    int countBefore, boolean finalExport) throws Exception {
        //Go through the list. If we have enough examples: remove the DataSet objects, merge and export them. Otherwise: do nothing
        int numExamples = 0;
        for (MultiDataSet ds : tempList) {
            numExamples += ds.getFeatures(0).size(0);
        }

        if (tempList.isEmpty() || (numExamples < minibatchSize && !finalExport)) {
            //No op
            return new Pair<>(countBefore, Collections.<String>emptyList());
        }

        List<String> exportPaths = new ArrayList<>();

        int countAfter = countBefore;

        //Batch the required number together
        int countSoFar = 0;
        List<MultiDataSet> tempToMerge = new ArrayList<>();
        while (!tempList.isEmpty() && countSoFar != minibatchSize) {
            MultiDataSet next = tempList.removeFirst();
            if (countSoFar + next.getFeatures(0).size(0) <= minibatchSize) {
                //Add the entire DataSet object
                tempToMerge.add(next);
                countSoFar += next.getFeatures(0).size(0);
            } else {
                //Split the DataSet
                List<MultiDataSet> examples = next.asList();
                for (MultiDataSet ds : examples) {
                    tempList.addFirst(ds);
                }
            }
        }
        //At this point: we should have the required number of examples in tempToMerge (unless it's a final export)
        MultiDataSet toExport = org.nd4j.linalg.dataset.MultiDataSet.merge(tempToMerge);
        exportPaths.add(export(toExport, partitionIdx, countAfter++));

        return new Pair<>(countAfter, exportPaths);
    }

    private String export(MultiDataSet dataSet, int partitionIdx, int outputCount) throws Exception {
        String filename = "mds_" + partitionIdx + jvmuid + "_" + outputCount + ".bin";

        URI uri = new URI(exportBaseDirectory
                        + (exportBaseDirectory.endsWith("/") || exportBaseDirectory.endsWith("\\") ? "" : "/")
                        + filename);
        FileSystem file = FileSystem.get(uri, conf);
        try (FSDataOutputStream out = file.create(new Path(uri))) {
            dataSet.save(out);
        }

        return uri.getPath();
    }
}
