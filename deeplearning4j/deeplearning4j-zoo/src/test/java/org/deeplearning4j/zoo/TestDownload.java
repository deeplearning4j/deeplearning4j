package org.deeplearning4j.zoo;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.model.NASNet;
import org.deeplearning4j.zoo.model.SimpleCNN;
import org.deeplearning4j.zoo.model.UNet;
import org.deeplearning4j.zoo.util.darknet.COCOLabels;
import org.deeplearning4j.zoo.util.darknet.DarknetLabels;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Tests downloads and checksum verification.
 *
 * @note This test uses a temporary directory, so local model copies won't be impacted
 * @author Justin Long (crockpotveggies)
 */
@Slf4j
public class TestDownload extends BaseDL4JTest {

    @ClassRule
    public static TemporaryFolder testDir = new TemporaryFolder();
    private static File f;

    @BeforeClass
    public static void before() throws Exception {
        f = testDir.newFolder();
        DL4JResources.setBaseDirectory(f);
    }

    @AfterClass
    public static void after(){
        DL4JResources.resetBaseDirectoryLocation();
    }

    @Test
    public void testDownloadAllModels() throws Exception {

        // iterate through each available model
        ZooModel[] models = new ZooModel[]{
                LeNet.builder().build(),
                SimpleCNN.builder().build(),
                UNet.builder().build(),
                NASNet.builder().build()
        };


        for (int i = 0; i < models.length; i++) {
            log.info("Testing zoo model " + models[i].getClass().getName());
            ZooModel model = models[i];

            for (PretrainedType pretrainedType : PretrainedType.values()) {
                if (model.pretrainedAvailable(pretrainedType)) {
                    model.initPretrained(pretrainedType);
                }
            }

            // clean up for current model
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            System.gc();
            Thread.sleep(1000);
        }
    }


    @Test
    public void testLabelsDownload() throws Exception {
        assertEquals("person", new COCOLabels().getLabel(0));
        assertEquals("kit fox", new DarknetLabels(true).getLabel(0));
        assertEquals("n02119789", new DarknetLabels(false).getLabel(0));
        assertEquals("tench", new ImageNetLabels().getLabel(0));
    }
}
