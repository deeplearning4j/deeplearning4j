package org.nd4j.autodiff.samediff.listeners;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.listeners.profiler.ProfilingListener;
import org.nd4j.autodiff.listeners.profiler.comparison.ProfileAnalyzer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class ProfilingListenerTest extends BaseNd4jTest {

    public ProfilingListenerTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testProfilingListenerSimple() throws Exception {

        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 3);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 1, 2);
        SDVariable w = sd.var("w", Nd4j.rand(DataType.FLOAT, 3, 2));
        SDVariable b = sd.var("b", Nd4j.rand(DataType.FLOAT, 1, 2));
        SDVariable sm = sd.nn.softmax("predictions", in.mmul("matmul", w).add("addbias", b));
        SDVariable loss = sd.loss.logLoss("loss", label, sm);

        INDArray i = Nd4j.rand(DataType.FLOAT, 1, 3);
        INDArray l = Nd4j.rand(DataType.FLOAT, 1, 2);


        File dir = testDir.newFolder();
        File f = new File(dir, "test.json");
        ProfilingListener listener = ProfilingListener.builder(f)
                .recordAll()
                .warmup(5)
                .build();

        sd.setListeners(listener);

        Map<String,INDArray> ph = new HashMap<>();
        ph.put("in", i);

        for( int x=0; x<10; x++ ) {
            sd.outputSingle(ph, "predictions");
        }

        String content = FileUtils.readFileToString(f, StandardCharsets.UTF_8);
        System.out.println(content);

        //Should be 2 begins and 2 ends for each entry
        //5 warmup iterations, 5 profile iterations, x2 for both the op name and the op "instance" name
        String[] opNames = {"mmul", "add", "softmax"};
        for(String s : opNames){
            assertEquals(s, 10, StringUtils.countMatches(content, s));
        }


        System.out.println("///////////////////////////////////////////");
        ProfileAnalyzer.summarizeProfile(f, ProfileAnalyzer.ProfileFormat.SAMEDIFF);

    }

    /*
    @Test
    public void testLoadTfProfile(){
        File f = new File("C:\\Temp\\sd_profiler\\tf_profile.json");
        ProfileAnalyzer.summarizeProfile(f, ProfileAnalyzer.ProfileFormat.TENSORFLOW);
    }

    @Test
    public void testLoadTfProfileDir(){
        File f = new File("C:\\Temp\\sd_profiler\\tf_multiple_profiles");
        ProfileAnalyzer.summarizeProfileDirectory(f, ProfileAnalyzer.ProfileFormat.TENSORFLOW);
    }

    @Test
    public void testLoadTfProfileDir2(){
        File f = new File("C:\\DL4J\\Git\\dl4j-dev-tools\\import-tests\\profiling\\mobilenet_v2_1.0_224_batch32_tf-1.15.0");
        ProfileAnalyzer.summarizeProfileDirectory(f, ProfileAnalyzer.ProfileFormat.TENSORFLOW);
    }
    */
}
