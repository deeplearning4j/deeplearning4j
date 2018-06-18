package org.deeplearning4j.optimizer.listener;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.checkpoint.CheckpointListener;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

public class TestCheckpointListener extends BaseDL4JTest {

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();

    private static Pair<MultiLayerNetwork,DataSetIterator> getNetAndData(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSetIterator iter = new IrisDataSetIterator(75,150);

        return new Pair<>(net, iter);
    }

    @Test
    public void testCheckpointListenerEvery2Epochs() throws Exception {
        File f = tempDir.newFolder();
        Pair<MultiLayerNetwork, DataSetIterator> p = getNetAndData();
        MultiLayerNetwork net = p.getFirst();
        DataSetIterator iter = p.getSecond();


        CheckpointListener l = new CheckpointListener.Builder(f)
                .keepAll()
                .saveEveryNEpochs(2)
                .build();
        net.setListeners(l);

        for(int i=0; i<10; i++ ){
            net.fit(iter);

            if(i > 0 && i % 2 == 0){
                assertEquals(1 + i/2, f.list().length);
            }
        }

        //Expect models saved at end of epochs: 1, 3, 5, 7, 9... (i.e., after 2, 4, 6 etc epochs)
        File[] files = f.listFiles();
        int count = 0;
        for(File f2 : files){
            if(!f2.getPath().endsWith(".zip")){
                continue;
            }

            int prefixLength = "checkpoint_".length();
            int num = Integer.parseInt(f2.getName().substring(prefixLength, prefixLength+1));

            MultiLayerNetwork n = ModelSerializer.restoreMultiLayerNetwork(f2, true);
            int expEpoch = 2 * (num + 1) - 1;   //Saved at the end of the previous epoch
            int expIter = (expEpoch+1) * 2;     //+1 due to epochs being zero indexed

            assertEquals(expEpoch, n.getEpochCount());
            assertEquals(expIter, n.getIterationCount());
            count++;
        }

        assertEquals(5, count);
        assertEquals(5, l.availableCheckpoints().size());
    }

    @Test
    public void testCheckpointListenerEvery5Iter() throws Exception {
        File f = tempDir.newFolder();
        Pair<MultiLayerNetwork, DataSetIterator> p = getNetAndData();
        MultiLayerNetwork net = p.getFirst();
        DataSetIterator iter = p.getSecond();


        CheckpointListener l = new CheckpointListener.Builder(f)
                .keepLast(3)
                .saveEveryNIterations(5)
                .build();
        net.setListeners(l);

        for(int i=0; i<20; i++ ){   //40 iterations total
            net.fit(iter);
        }

        //Expect models saved at iterations: 5, 10, 15, 20, 25, 30, 35  (training does 0 to 39 here)
        //But: keep only 25, 30, 35
        File[] files = f.listFiles();
        int count = 0;
        Set<Integer> ns = new HashSet<>();
        for(File f2 : files){
            if(!f2.getPath().endsWith(".zip")){
                continue;
            }
            count++;
            int prefixLength = "checkpoint_".length();
            int num = Integer.parseInt(f2.getName().substring(prefixLength, prefixLength+1));

            MultiLayerNetwork n = ModelSerializer.restoreMultiLayerNetwork(f2, true);
            int expIter = 5 * (num+1);
            assertEquals(expIter, n.getIterationCount());

            ns.add(n.getIterationCount());
            count++;
        }

        assertEquals(ns.toString(), 3, ns.size());
        assertTrue(ns.contains(25));
        assertTrue(ns.contains(30));
        assertTrue(ns.contains(35));

        assertEquals(3, l.availableCheckpoints().size());
    }

    @Test
    public void testCheckpointListenerEveryTimeUnit() throws Exception {
        File f = tempDir.newFolder();
        Pair<MultiLayerNetwork, DataSetIterator> p = getNetAndData();
        MultiLayerNetwork net = p.getFirst();
        DataSetIterator iter = p.getSecond();


        CheckpointListener l = new CheckpointListener.Builder(f)
                .keepLast(3)
                .saveEvery(3, TimeUnit.SECONDS)
                .build();
        net.setListeners(l);

        for(int i=0; i<5; i++ ){   //10 iterations total
            net.fit(iter);
            Thread.sleep(4000);
        }

        //Expect models saved at iterations: 2, 4, 6, 8 (iterations 0 and 1 shoud happen before first 3 seconds is up)
        //But: keep only 5, 7, 9
        File[] files = f.listFiles();
        Set<Integer> ns = new HashSet<>();
        for(File f2 : files){
            if(!f2.getPath().endsWith(".zip")){
                continue;
            }

            int prefixLength = "checkpoint_".length();
            int num = Integer.parseInt(f2.getName().substring(prefixLength, prefixLength+1));

            MultiLayerNetwork n = ModelSerializer.restoreMultiLayerNetwork(f2, true);
            int expIter = 2 * (num + 1);
            assertEquals(expIter, n.getIterationCount());

            ns.add(n.getIterationCount());
        }

        assertEquals(3, l.availableCheckpoints().size());
        assertEquals(ns.toString(), 3, ns.size());
        assertTrue(ns.containsAll(Arrays.asList(4,6,8)));
    }

    @Test
    public void testCheckpointListenerKeepLast3AndEvery3() throws Exception {
        File f = tempDir.newFolder();
        Pair<MultiLayerNetwork, DataSetIterator> p = getNetAndData();
        MultiLayerNetwork net = p.getFirst();
        DataSetIterator iter = p.getSecond();


        CheckpointListener l = new CheckpointListener.Builder(f)
                .keepLastAndEvery(3, 3)
                .saveEveryNEpochs(2)
                .build();
        net.setListeners(l);

        for(int i=0; i<20; i++ ){   //40 iterations total
            net.fit(iter);
        }

        //Expect models saved at end of epochs: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        //But: keep only 5, 11, 15, 17, 19
        File[] files = f.listFiles();
        int count = 0;
        Set<Integer> ns = new HashSet<>();
        for(File f2 : files){
            if(!f2.getPath().endsWith(".zip")){
                continue;
            }
            count++;
            int prefixLength = "checkpoint_".length();
            int end = f2.getName().lastIndexOf("_");
            int num = Integer.parseInt(f2.getName().substring(prefixLength, end));

            MultiLayerNetwork n = ModelSerializer.restoreMultiLayerNetwork(f2, true);
            int expEpoch = 2 * (num+1) - 1;
            assertEquals(expEpoch, n.getEpochCount());

            ns.add(n.getEpochCount());
            count++;
        }

        assertEquals(ns.toString(), 5, ns.size());
        assertTrue(ns.toString(), ns.containsAll(Arrays.asList(5, 11, 15, 17, 19)));

        assertEquals(5, l.availableCheckpoints().size());
    }

    @Test
    public void testDeleteExisting() throws Exception {
        File f = tempDir.newFolder();
        Pair<MultiLayerNetwork, DataSetIterator> p = getNetAndData();
        MultiLayerNetwork net = p.getFirst();
        DataSetIterator iter = p.getSecond();


        CheckpointListener l = new CheckpointListener.Builder(f)
                .keepAll()
                .saveEveryNEpochs(1)
                .build();
        net.setListeners(l);

        for(int i=0; i<3; i++ ){
            net.fit(iter);
        }

        //Now, create new listener:
        try{
            l = new CheckpointListener.Builder(f)
                    .keepAll()
                    .saveEveryNEpochs(1)
                    .build();
            fail("Expected exception");
        } catch (IllegalStateException e){
            assertTrue(e.getMessage().contains("Use deleteExisting(true)"));
        }

        l = new CheckpointListener.Builder(f)
                .keepAll()
                .saveEveryNEpochs(1)
                .deleteExisting(true)
                .build();
        net.setListeners(l);

        net.fit(iter);

        File[] fList = f.listFiles();   //checkpoint meta file + 1 checkpoint
        assertNotNull(fList);
        assertEquals(2, fList.length);
    }
}
