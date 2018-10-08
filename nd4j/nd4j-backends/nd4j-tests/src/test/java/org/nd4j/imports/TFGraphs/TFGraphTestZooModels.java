package org.nd4j.imports.TFGraphs;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.OpValidationSuite;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.BiFunction;
import org.nd4j.resources.Downloader;
import org.nd4j.util.ArchiveUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@RunWith(Parameterized.class)
@Slf4j
public class TFGraphTestZooModels {

    @ClassRule
    public static TemporaryFolder classTestDir = new TemporaryFolder();

    public static final String[] IGNORE_REGEXES = {
            //https://github.com/deeplearning4j/deeplearning4j/issues/6397
            "deeplab.*",

            //https://github.com/deeplearning4j/deeplearning4j/issues/6462
            "inception_v4_2018_04_27",
            "inception_resnet_v2_2018_04_27"

    };

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();
    private static File currentTestDir;

    public static final File BASE_MODEL_DL_DIR = new File(System.getProperty("user.home"), ".nd4jtests");

    private static final String BASE_DIR = "tf_graphs/zoo_models";
    private static final String MODEL_FILENAME = "tf_model.txt";

    private Map<String, INDArray> inputs;
    private Map<String, INDArray> predictions;
    private String modelName;
    private File localTestDir;

    public static final BiFunction<File,String,SameDiff> LOADER = new RemoteCachingLoader();

    public static class RemoteCachingLoader implements BiFunction<File,String,SameDiff> {
        @Override
        public SameDiff apply(File file, String name) {
            try {
                String s = FileUtils.readFileToString(file, StandardCharsets.UTF_8).replaceAll("\r\n","\n");
                String[] split = s.split("\n");
                if(split.length != 2){
                    throw new IllegalStateException("Invalid file: expected 2 lines with URL and MD5 hash. Got " + split.length + " lines");
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
                    for(String f : files){
                        if(f.endsWith(".pb")){
                            if(toExtract != null){
                                throw new IllegalStateException("Found multiple .pb files in archive: " + toExtract + " and " + f);
                            }
                            toExtract = f;
                        }
                    }
                    Preconditions.checkState(toExtract != null, "Found to .pb files in archive: %s", localFile.getAbsolutePath());

                    Preconditions.checkNotNull(currentTestDir, "currentTestDir has not been set (is null)");
                    modelFile = new File(currentTestDir, "tf_model.pb");
                    ArchiveUtils.tarGzExtractSingleFile(localFile, modelFile, toExtract);
                } else if(filename.endsWith(".zip")){
                    throw new IllegalStateException("Not yet implemented");
                } else {
                    throw new IllegalStateException("Unknown format: " + filename);
                }

                return TFGraphTestAllHelper.LOADER.apply(modelFile, name);
            } catch (IOException e){
                throw new RuntimeException(e);
            }
        }
    }

    @BeforeClass
    public static void beforeClass(){
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @Parameterized.Parameters(name="{2}")
    public static Collection<Object[]> data() throws IOException {
        classTestDir.create();
        File baseDir = classTestDir.newFolder();    // new File(System.getProperty("java.io.tmpdir"), UUID.randomUUID().toString());
        List<Object[]> params = TFGraphTestAllHelper.fetchTestParams(BASE_DIR, MODEL_FILENAME, TFGraphTestAllHelper.ExecuteWith.SAMEDIFF, baseDir);
        return params;
    }

    public TFGraphTestZooModels(Map<String, INDArray> inputs, Map<String, INDArray> predictions, String modelName, File localTestDir) throws IOException {
        this.inputs = inputs;
        this.predictions = predictions;
        this.modelName = modelName;
        this.localTestDir = localTestDir;
    }

    @Test   //(timeout = 360000L)
    public void testOutputOnly() throws Exception {
//        if(!modelName.equals("mobilenet_v1_0.5_128")){
//        if(!modelName.equals("nasnet_mobile_2018_04_27")){
//        if(!modelName.equals("resnetv2_imagenet_frozen_graph")){
//        if(!modelName.equals("mobilenet_v2_1.0_224")){
//        if(!modelName.equals("densenet_2018_04_27")){
//        if(!modelName.equals("inception_resnet_v2_2018_04_27")){
//            OpValidationSuite.ignoreFailing();
//        }
        currentTestDir = testDir.newFolder();

        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.NAN_PANIC);

        Nd4j.create(1);
        for(String s : IGNORE_REGEXES){
            if(modelName.matches(s)){
                log.info("\n\tIGNORE MODEL ON REGEX: {} - regex {}", modelName, s);
                OpValidationSuite.ignoreFailing();
            }
        }

        Double maxRE = 1e-3;
        Double minAbs = 1e-4;
        currentTestDir = testDir.newFolder();
        TFGraphTestAllHelper.checkOnlyOutput(inputs, predictions, modelName, BASE_DIR, MODEL_FILENAME, TFGraphTestAllHelper.ExecuteWith.SAMEDIFF,
                LOADER, maxRE, minAbs);


//        Double maxRE = 1e-2;
//        Double minAbs = 1e-3;
//        TFGraphTestAllHelper.checkIntermediate(inputs, modelName, BASE_DIR, MODEL_FILENAME, TFGraphTestAllHelper.ExecuteWith.SAMEDIFF,
//                LOADER, maxRE, minAbs, localTestDir);
    }
}
