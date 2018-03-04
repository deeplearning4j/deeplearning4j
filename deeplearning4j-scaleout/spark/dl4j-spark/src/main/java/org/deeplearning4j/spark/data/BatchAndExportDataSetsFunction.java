/*-
 *  * Copyright 2016 Skymind,Inc.
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
 */

package org.deeplearning4j.spark.data;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.util.UIDProvider;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.primitives.Pair;

import java.net.URI;
import java.util.*;

/**
 * Function used with {@code RDD<DataSet>.mapPartitionsWithIndex}.
 * It does two things:
 * 1. Batch DataSets together, to the specified minibatch size. This may result in splitting or combining existing
 * DataSet objects as required
 * 2. Export the DataSet objects to the specified directory.
 * <p>
 * Naming convention for exported files:
 * "dataset_" + partitionIdx + JVM_UID + "_" + idx + ".bin"
 * where 'idx' is the index of the DataSet objects in this partition
 *
 * @author Alex Black
 */
public class BatchAndExportDataSetsFunction implements Function2<Integer, Iterator<DataSet>, Iterator<String>> {

    private static final Configuration conf = new Configuration();
    private final int minibatchSize;
    private final String exportBaseDirectory;
    private final String jvmuid;

    /**
     * @param minibatchSize       Minibatch size to combine examples to (if necessary)
     * @param exportBaseDirectory Base directory for exporting
     */
    public BatchAndExportDataSetsFunction(int minibatchSize, String exportBaseDirectory) {
        this.minibatchSize = minibatchSize;
        this.exportBaseDirectory = exportBaseDirectory;
        String fullUID = UIDProvider.getJVMUID();
        this.jvmuid = (fullUID.length() <= 8 ? fullUID : fullUID.substring(0, 8));
    }

    @Override
    public Iterator<String> call(Integer partitionIdx, Iterator<DataSet> iterator) throws Exception {

        List<String> outputPaths = new ArrayList<>();
        LinkedList<DataSet> tempList = new LinkedList<>();

        int count = 0;
        while (iterator.hasNext()) {
            DataSet next = iterator.next();
            if (next.numExamples() == minibatchSize) {
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

    private Pair<Integer, List<String>> processList(LinkedList<DataSet> tempList, int partitionIdx, int countBefore,
                    boolean finalExport) throws Exception {
        //Go through the list. If we have enough examples: remove the DataSet objects, merge and export them. Otherwise: do nothing
        int numExamples = 0;
        for (DataSet ds : tempList) {
            numExamples += ds.numExamples();
        }

        if (tempList.isEmpty() || (numExamples < minibatchSize && !finalExport)) {
            //No op
            return new Pair<>(countBefore, Collections.<String>emptyList());
        }

        List<String> exportPaths = new ArrayList<>();

        int countAfter = countBefore;

        //Batch the required number together
        int countSoFar = 0;
        List<DataSet> tempToMerge = new ArrayList<>();
        while (!tempList.isEmpty() && countSoFar != minibatchSize) {
            DataSet next = tempList.removeFirst();
            if (countSoFar + next.numExamples() <= minibatchSize) {
                //Add the entire DataSet object
                tempToMerge.add(next);
                countSoFar += next.numExamples();
            } else {
                //Split the DataSet
                List<DataSet> examples = next.asList();
                for (DataSet ds : examples) {
                    tempList.addFirst(ds);
                }
            }
        }
        //At this point: we should have the required number of examples in tempToMerge (unless it's a final export)
        DataSet toExport = DataSet.merge(tempToMerge);
        exportPaths.add(export(toExport, partitionIdx, countAfter++));

        return new Pair<>(countAfter, exportPaths);
    }

    private String export(DataSet dataSet, int partitionIdx, int outputCount) throws Exception {
        String filename = "dataset_" + partitionIdx + jvmuid + "_" + outputCount + ".bin";

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
