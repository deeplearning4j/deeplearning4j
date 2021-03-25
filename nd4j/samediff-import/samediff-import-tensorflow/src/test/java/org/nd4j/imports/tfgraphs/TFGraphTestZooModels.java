/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.imports.tfgraphs;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.io.TempDir;


import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.function.BiFunction;
import org.nd4j.common.resources.Downloader;
import org.nd4j.common.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;


@Slf4j
@Tag(TagNames.LONG_TEST)
@Tag(TagNames.LARGE_RESOURCES)
public class TFGraphTestZooModels { //Note: Can't extend BaseNd4jTest here as we need no-arg constructor for parameterized tests
    @TempDir
    static Path classTestDir;


    public static final String[] IGNORE_REGEXES = {
            //2019/07/22 - Result value failure
            "xlnet_cased_L-24_H-1024_A-16",

            // 2019/07/22 - OOM, Passes with sufficient memory (16GB heap, 32GB off-heap tested)
            "deeplabv3_xception_ade20k_train",

            //2019/07/03 - o.n.i.g.t.TFGraphMapper - No TensorFlow descriptor found for tensor "sample_sequence/model/h0/attn/MatMul", op "BatchMatMulV2"
            //org.nd4j.linalg.exception.ND4JIllegalStateException: No tensorflow op found for Multinomial possibly missing operation class?
            // @ TFGraphMapper.mapNodeType(TFGraphMapper.java:593)
            // Missing Multinormal op, see https://github.com/eclipse/deeplearning4j/issues/7913
            "gpt-2_117M",

            //AB 2020/01/08, all 3 - https://github.com/eclipse/deeplearning4j/issues/8603
            "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03",
            "ssd_mobilenet_v1_coco_2018_01_28",
            "faster_rcnn_resnet101_coco_2018_01_28",

            //2019/06/24 - JVM crash on linux-x86_64-cpu-avx2 and -avx512 CI machines only - runs fine elsewhere
            "deeplabv3_pascal_train_aug_2018_01_04",
    };

    public static final String[] IGNORE_REGEXES_LIBND4J_EXEC_ONLY = {

            // Graph wasn't topsorted for all Keras RNNs (possible TF's too)
            // https://github.com/eclipse/deeplearning4j/issues/7974
            //Alexnet takes too long
            "PorV-RNN",
            "temperature_bidirectional_63",
            "temperature_stacked_63",
            "text_gen_81",


            // 2019/05/20 - Buffer is too big to export? https://github.com/eclipse/deeplearning4j/issues/7760
            // File: C:/DL4J/Git/deeplearning4j/libnd4j/blasbuild/cpu/flatbuffers-src/include/flatbuffers/flatbuffers.h, Line 668
            //Expression: size() < FLATBUFFERS_MAX_BUFFER_SIZE
            "deeplabv3_pascal_train_aug_2018_01_04"
    };


    public static File currentTestDir;

    public static final File BASE_MODEL_DL_DIR = new File(getBaseModelDir(), ".nd4jtests");

    private static final String BASE_DIR = "tf_graphs/zoo_models";
    private static final String MODEL_FILENAME = "tf_model.txt";

    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    private File localTestDir;

    public static String getBaseModelDir(){
        String s = System.getProperty("org.nd4j.tests.modeldir");
        if(s != null && !s.isEmpty()){
            return s;
        }
        return System.getProperty("user.home");
    }

    public static final BiFunction<File,String,SameDiff> LOADER = new RemoteCachingLoader();

    public static class RemoteCachingLoader implements BiFunction<File,String,SameDiff> {
        @Override
        public SameDiff apply(File file, String name) {
            try {
                String s = FileUtils.readFileToString(file, StandardCharsets.UTF_8).replaceAll("\r\n","\n");
                String[] split = s.split("\n");
                if(split.length != 2 && split.length != 3){
                    throw new IllegalStateException("Invalid file: expected 2 lines with URL and MD5 hash, or 3 lines with " +
                            "URL, MD5 hash and file name. Got " + split.length + " lines");
                }
                String url = split[0];
                String md5 = split[1];

                File localDir = new File(BASE_MODEL_DL_DIR, name);
                if(!localDir.exists())
                    localDir.mkdirs();

                String filename = FilenameUtils.getName(url);
                File localFile = new File(localDir, filename);

                if(localFile.exists() && !Downloader.checkMD5OfFile(md5, localFile)) {
                    log.info("Deleting local file: does not match MD5. {}", localFile.getAbsolutePath());
                    localFile.delete();
                }

                if (!localFile.exists()) {
                    log.info("Starting resource download from: {} to {}", url, localFile.getAbsolutePath());
                    Downloader.download(name, new URL(url), localFile, md5, 3);
                }

                File modelFile;

                if(filename.endsWith(".pb")) {
                    modelFile = localFile;
                } else if(filename.endsWith(".tar.gz") || filename.endsWith(".tgz")){
                    List<String> files = ArchiveUtils.tarGzListFiles(localFile);
                    String toExtract = null;
                    if(split.length == 3){
                        //Extract specific file
                        toExtract = split[2];
                    } else {
                        List<String> pbFiles = new ArrayList<>();
                        for (String f : files) {
                            if (f.endsWith(".pb")) {
                                pbFiles.add(f);
                            }
                        }

                        if(pbFiles.size() == 1){
                            toExtract = pbFiles.get(0);
                        } else if(pbFiles.size() == 0){
                            toExtract = null;
                        } else {
                            //Multiple files... try to find "frozen_inference_graph.pb"
                            for(String str : pbFiles){
                                if(str.endsWith("frozen_inference_graph.pb")) {
                                    toExtract = str;
                                }
                            }
                            if(toExtract == null){
                                throw new IllegalStateException("Found multiple .pb files in archive: " + localFile + " - pb files in archive: " + pbFiles);
                            }
                        }
                    }
                    Preconditions.checkState(toExtract != null, "Found no .pb files in archive: %s", localFile.getAbsolutePath());

                    Preconditions.checkNotNull(currentTestDir, "currentTestDir has not been set (is null)");
                    modelFile = new File(currentTestDir, "tf_model.pb");
                    ArchiveUtils.tarGzExtractSingleFile(localFile, modelFile, toExtract);
                } else if(filename.endsWith(".zip")){
                    throw new IllegalStateException("ZIP support - not yet implemented");
                } else {
                    throw new IllegalStateException("Unknown format: " + filename);
                }

                SameDiff apply = TFGraphTestAllHelper.LOADER.apply(modelFile, name);
                //"suggest" a GC before running the model to mitigate OOM
                System.gc();
                return apply;
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }

    @BeforeAll
    public static void beforeClass(){
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
    }

    public static Stream<Arguments> data() throws IOException {
        classTestDir.toFile().mkdir();
        File baseDir = classTestDir.toFile();    // new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString());
        List<Object[]> params = TFGraphTestAllHelper.fetchTestParams(BASE_DIR, MODEL_FILENAME, TFGraphTestAllHelper.ExecuteWith.SAMEDIFF, baseDir);
        return params.stream().map(Arguments::of);
    }

    private static Boolean isPPC = null;

    public static boolean isPPC(){
        if(isPPC == null){
            ///mnt/jenkins/workspace/deeplearning4j-bugfix-tests-linux-ppc64le-cpu/
            File f = new File("");
            String path = f.getAbsolutePath();
            log.info("Current directory: {}", path);
            isPPC = path.contains("ppc64le");
        }
        return isPPC;
    }

    @Test   //(timeout = 360000L)
    @ParameterizedTest
    @MethodSource("#data")
    public void testOutputOnly(@TempDir Path testDir) throws Exception {
        if(isPPC()){
            /*
            Ugly hack to temporarily disable tests on PPC only on CI
            Issue logged here: https://github.com/eclipse/deeplearning4j/issues/7657
            These will be re-enabled for PPC once fixed - in the mean time, remaining tests will be used to detect and prevent regressions
             */

            log.warn("TEMPORARILY SKIPPING TEST ON PPC ARCHITECTURE DUE TO KNOWN JVM CRASH ISSUES - SEE https://github.com/eclipse/deeplearning4j/issues/7657");
            //OpValidationSuite.ignoreFailing();
        }

//        if(!modelName.startsWith("ssd_mobilenet_v1_coco_2018_01_28")){
//        if(!modelName.startsWith("ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03")){
//        if(!modelName.startsWith("faster_rcnn_resnet101_coco_2018_01_28")){
//            OpValidationSuite.ignoreFailing();
//        }
        currentTestDir = testDir.toFile();

//        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.NAN_PANIC);
        Nd4j.getMemoryManager().setAutoGcWindow(2000);

        Nd4j.create(1);
        if(ArrayUtils.contains(IGNORE_REGEXES, modelName)){
            log.info("\n\tIGNORE MODEL ON REGEX: {} - regex {}", modelName, modelName);
           // OpValidationSuite.ignoreFailing();
        }

        Double maxRE = 1e-3;
        Double minAbs = 1e-4;
        currentTestDir = testDir.toFile();
        log.info("----- SameDiff Exec: {} -----", modelName);
        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, BASE_DIR, MODEL_FILENAME, TFGraphTestAllHelper.ExecuteWith.SAMEDIFF,
                LOADER, maxRE, minAbs, false);

        if(ArrayUtils.contains(IGNORE_REGEXES_LIBND4J_EXEC_ONLY, modelName)){
            log.warn("\n\tIGNORING MODEL FOR LIBND4J EXECUTION ONLY: ");
            return;
        }

        //Libnd4j exec:
        /*
        //AB 2019/10/19 - Libnd4j execution disabled pending execution rewrite
        currentTestDir = testDir.newFolder();
        log.info("----- Libnd4j Exec: {} -----", modelName);
        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, BASE_DIR, MODEL_FILENAME, TFGraphTestAllHelper.ExecuteWith.LIBND4J,
                LOADER, maxRE, minAbs);
         */
    }
}
