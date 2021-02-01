/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.listeners;

import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.autodiff.listeners.checkpoint.CheckpointListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.IrisDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

public class CheckpointListenerTest extends BaseNd4jTest {

    public CheckpointListenerTest(Nd4jBackend backend){
        super(backend);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    public static SameDiff getModel(){
        Nd4j.getRandom().setSeed(12345);
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, -1, 4);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, 3);
        SDVariable w = sd.var("W", Nd4j.rand(DataType.FLOAT, 4, 3));
        SDVariable b = sd.var("b", DataType.FLOAT, 3);

        SDVariable mmul = in.mmul(w).add(b);
        SDVariable softmax = sd.nn().softmax(mmul);
        SDVariable loss = sd.loss().logLoss("loss", label, softmax);

        sd.setTrainingConfig(TrainingConfig.builder()
                .dataSetFeatureMapping("in")
                .dataSetLabelMapping("label")
                .updater(new Adam(1e-2))
                .weightDecay(1e-2, true)
                .build());

        return sd;
    }

    public static DataSetIterator getIter() {
        return getIter(15, 150);
    }

    public static DataSetIterator getIter(int batch, int totalExamples){
        return new IrisDataSetIterator(batch, totalExamples);
    }


    @Test
    public void testCheckpointEveryEpoch() throws Exception {
        File dir = testDir.newFolder();

        SameDiff sd = getModel();
        CheckpointListener l = CheckpointListener.builder(dir)
                .saveEveryNEpochs(1)
                .build();

        sd.setListeners(l);

        DataSetIterator iter = getIter();
        sd.fit(iter, 3);

        File[] files = dir.listFiles();
        String s1 = "checkpoint-0_epoch-0_iter-9";      //Note: epoch is 10 iterations, 0-9, 10-19, 20-29, etc
        String s2 = "checkpoint-1_epoch-1_iter-19";
        String s3 = "checkpoint-2_epoch-2_iter-29";
        boolean found1 = false;
        boolean found2 = false;
        boolean found3 = false;
        for(File f : files){
            String s = f.getAbsolutePath();
            if(s.contains(s1))
                found1 = true;
            if(s.contains(s2))
                found2 = true;
            if(s.contains(s3))
                found3 = true;
        }
        assertEquals(4, files.length);  //3 checkpoints and 1 text file (metadata)
        assertTrue(found1 && found2 && found3);
    }

    @Test
    public void testCheckpointEvery5Iter() throws Exception {
        File dir = testDir.newFolder();

        SameDiff sd = getModel();
        CheckpointListener l = CheckpointListener.builder(dir)
                .saveEveryNIterations(5)
                .build();

        sd.setListeners(l);

        DataSetIterator iter = getIter();
        sd.fit(iter, 2);                        //2 epochs = 20 iter

        File[] files = dir.listFiles();
        List<String> names = Arrays.asList(
                "checkpoint-0_epoch-0_iter-4",
                "checkpoint-1_epoch-0_iter-9",
                "checkpoint-2_epoch-1_iter-14",
                "checkpoint-3_epoch-1_iter-19");
        boolean[] found = new boolean[names.size()];
        for(File f : files){
            String s = f.getAbsolutePath();
//            System.out.println(s);
            for( int i=0; i<names.size(); i++ ){
                if(s.contains(names.get(i))){
                    found[i] = true;
                    break;
                }
            }
        }
        assertEquals(5, files.length);  //4 checkpoints and 1 text file (metadata)

        for( int i=0; i<found.length; i++ ){
            assertTrue(names.get(i), found[i]);
        }
    }


    @Test
    public void testCheckpointListenerEveryTimeUnit() throws Exception {
        File dir = testDir.newFolder();
        SameDiff sd = getModel();

        CheckpointListener l = new CheckpointListener.Builder(dir)
                .keepLast(2)
                .saveEvery(4, TimeUnit.SECONDS)
                .build();
        sd.setListeners(l);

        DataSetIterator iter = getIter(15, 150);

        for(int i=0; i<5; i++ ){   //10 iterations total
            sd.fit(iter, 1);
            Thread.sleep(5000);
        }

        //Expect models saved at iterations: 10, 20, 30, 40
        //But: keep only 30, 40
        File[] files = dir.listFiles();

        assertEquals(3, files.length);  //2 files, 1 metadata file

        List<String> names = Arrays.asList(
                "checkpoint-2_epoch-3_iter-30",
                "checkpoint-3_epoch-4_iter-40");
        boolean[] found = new boolean[names.size()];
        for(File f : files){
            String s = f.getAbsolutePath();
//            System.out.println(s);
            for( int i=0; i<names.size(); i++ ){
                if(s.contains(names.get(i))){
                    found[i] = true;
                    break;
                }
            }
        }

        for( int i=0; i<found.length; i++ ){
            assertTrue(names.get(i), found[i]);
        }
    }

    @Test
    public void testCheckpointListenerKeepLast3AndEvery3() throws Exception {
        File dir = testDir.newFolder();
        SameDiff sd = getModel();

        CheckpointListener l = new CheckpointListener.Builder(dir)
                .keepLastAndEvery(3, 3)
                .saveEveryNEpochs(2)
                .fileNamePrefix("myFilePrefix")
                .build();
        sd.setListeners(l);

        DataSetIterator iter = getIter();

        sd.fit(iter, 20);

        //Expect models saved at end of epochs: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        //But: keep only 5, 11, 15, 17, 19
        File[] files = dir.listFiles();
        int count = 0;
        Set<Integer> cpNums = new HashSet<>();
        Set<Integer> epochNums = new HashSet<>();
        for(File f2 : files){
            if(!f2.getPath().endsWith(".bin")){
                continue;
            }
            count++;
            int idx = f2.getName().indexOf("epoch-");
            int end = f2.getName().indexOf("_", idx);
            int num = Integer.parseInt(f2.getName().substring(idx + "epoch-".length(), end));
            epochNums.add(num);

            int start = f2.getName().indexOf("checkpoint-");
            end = f2.getName().indexOf("_", start + "checkpoint-".length());
            int epochNum = Integer.parseInt(f2.getName().substring(start + "checkpoint-".length(), end));
            cpNums.add(epochNum);
        }

        assertEquals(cpNums.toString(), 5, cpNums.size());
        Assert.assertTrue(cpNums.toString(), cpNums.containsAll(Arrays.asList(2, 5, 7, 8, 9)));
        Assert.assertTrue(epochNums.toString(), epochNums.containsAll(Arrays.asList(5, 11, 15, 17, 19)));

        assertEquals(5, l.availableCheckpoints().size());
    }
}
