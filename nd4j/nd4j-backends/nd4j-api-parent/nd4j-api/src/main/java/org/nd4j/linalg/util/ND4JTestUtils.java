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

package org.nd4j.linalg.util;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.io.FileUtils;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.BiFunction;
import org.nd4j.linalg.primitives.Triple;

import java.io.File;
import java.net.URI;
import java.util.*;

/**
 * Test utilitiez for ND4J
 *
 * @author Alex Black
 */
public class ND4JTestUtils {

    private ND4JTestUtils(){ }


    @AllArgsConstructor
    @Data
    public static class ComparisonResult {
        List<Triple<File,File,Boolean>> allResults;
        List<Triple<File,File,Boolean>> passed;
        List<Triple<File,File,Boolean>> failed;
        List<File> skippedDir1;
        List<File> skippedDir2;
    }

    /**
     * A function for use with {@link #validateSerializedArrays(File, File, boolean, BiFunction)} using {@link INDArray#equals(Object)}
     */
    public static class EqualsFn implements BiFunction<INDArray,INDArray,Boolean> {
        @Override
        public Boolean apply(INDArray i1, INDArray i2) {
            return i1.equals(i2);
        }
    }

    /**
     * A function for use with {@link #validateSerializedArrays(File, File, boolean, BiFunction)} using {@link INDArray#equalsWithEps(Object, double)}
     */
    @AllArgsConstructor
    public static class EqualsWithEpsFn implements BiFunction<INDArray,INDArray,Boolean> {
        private final double eps;

        @Override
        public Boolean apply(INDArray i1, INDArray i2) {
            return i1.equalsWithEps(i2, eps);
        }
    }

    /**
     * Scan the specified directories for matching files (i.e., same path relative to their respective root directories)
     * and compare the contents using INDArray.equals (via {@link EqualsFn}
     * Assumes the saved files represent INDArrays saved with {@link Nd4j#saveBinary(INDArray, File)}
     * @param dir1      First directory
     * @param dir2      Second directory
     * @param recursive Whether to search recursively (i.e., include files in subdirectories
     * @return Comparison results
     */
    public static ComparisonResult validateSerializedArrays(File dir1, File dir2, boolean recursive) throws Exception {
        return validateSerializedArrays(dir1, dir2, recursive, new EqualsFn());
    }

    /**
     * Scan the specified directories for matching files (i.e., same path relative to their respective root directories)
     * and compare the contents using a provided function.<br>
     * Assumes the saved files represent INDArrays saved with {@link Nd4j#saveBinary(INDArray, File)}
     * @param dir1      First directory
     * @param dir2      Second directory
     * @param recursive Whether to search recursively (i.e., include files in subdirectories
     * @return Comparison results
     */
    public static ComparisonResult validateSerializedArrays(File dir1, File dir2, boolean recursive, BiFunction<INDArray,INDArray,Boolean> evalFn) throws Exception {
        File[] f1 = FileUtils.listFiles(dir1, null, recursive).toArray(new File[0]);
        File[] f2 = FileUtils.listFiles(dir2, null, recursive).toArray(new File[0]);

        Preconditions.checkState(f1.length > 0, "No files found for directory 1: %s", dir1.getAbsolutePath() );
        Preconditions.checkState(f2.length > 0, "No files found for directory 2: %s", dir2.getAbsolutePath() );

        Map<String,File> relativized1 = new HashMap<>();
        Map<String,File> relativized2 = new HashMap<>();

        URI u = dir1.toURI();
        for(File f : f1){
            if(!f.isFile())
                continue;
            String relative = u.relativize(f.toURI()).getPath();
            relativized1.put(relative, f);
        }

        u = dir2.toURI();
        for(File f : f2){
            if(!f.isFile())
                continue;
            String relative = u.relativize(f.toURI()).getPath();
            relativized2.put(relative, f);
        }

        List<File> skipped1 = new ArrayList<>();
        for(String s : relativized1.keySet()){
            if(!relativized2.containsKey(s)){
                skipped1.add(relativized1.get(s));
            }
        }

        List<File> skipped2 = new ArrayList<>();
        for(String s : relativized2.keySet()){
            if(!relativized1.containsKey(s)){
                skipped2.add(relativized1.get(s));
            }
        }

        List<Triple<File,File,Boolean>> allResults = new ArrayList<>();
        List<Triple<File,File,Boolean>> passed = new ArrayList<>();
        List<Triple<File,File,Boolean>> failed = new ArrayList<>();
        for(Map.Entry<String,File> e : relativized1.entrySet()){
            File file1 = e.getValue();
            File file2 = relativized2.get(e.getKey());

            if(file2 == null)
                continue;

            INDArray i1 = Nd4j.readBinary(file1);
            INDArray i2 = Nd4j.readBinary(file2);
            boolean b = evalFn.apply(i1, i2);
            Triple<File,File,Boolean> t = new Triple<>(file1, file2, b);
            allResults.add(t);
            if(b){
                passed.add(t);
            } else {
                failed.add(t);
            }
        }

        Comparator<Triple<File,File,Boolean>> c = new Comparator<Triple<File, File, Boolean>>() {
            @Override
            public int compare(Triple<File, File, Boolean> o1, Triple<File, File, Boolean> o2) {
                return o1.getFirst().compareTo(o2.getFirst());
            }
        };

        Collections.sort(allResults, c);
        Collections.sort(passed, c);
        Collections.sort(failed, c);
        Collections.sort(skipped1);
        Collections.sort(skipped2);


        return new ComparisonResult(allResults, passed, failed, skipped1, skipped2);
    }
}
