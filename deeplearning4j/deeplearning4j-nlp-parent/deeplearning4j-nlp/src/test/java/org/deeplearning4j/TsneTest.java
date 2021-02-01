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

package org.deeplearning4j;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

@Slf4j
public class TsneTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 180000L;
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Override
    public DataType getDefaultFPDataType() {
        return DataType.FLOAT;
    }

    @Test
    public void testSimple() throws Exception {
        //Simple sanity check

        for( int test=0; test <=1; test++){
            boolean syntheticData = test == 1;
            WorkspaceMode wsm = test == 0 ? WorkspaceMode.NONE : WorkspaceMode.ENABLED;
            log.info("Starting test: WSM={}, syntheticData={}", wsm, syntheticData);

            //STEP 1: Initialization
            int iterations = 50;
            //create an n-dimensional array of doubles
            Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
            List<String> cacheList = new ArrayList<>(); //cacheList is a dynamic array of strings used to hold all words

            //STEP 2: Turn text input into a list of words
            INDArray weights;
            if(syntheticData){
                weights = Nd4j.rand(250, 200);
            } else {
                log.info("Load & Vectorize data....");
                File wordFile = new ClassPathResource("deeplearning4j-tsne/words.txt").getFile();   //Open the file
                //Get the data of all unique word vectors
                Pair<InMemoryLookupTable, VocabCache> vectors = WordVectorSerializer.loadTxt(wordFile);
                VocabCache cache = vectors.getSecond();
                weights = vectors.getFirst().getSyn0();    //seperate weights of unique words into their own list

                for (int i = 0; i < cache.numWords(); i++)   //seperate strings of words into their own list
                    cacheList.add(cache.wordAtIndex(i));
            }

            //STEP 3: build a dual-tree tsne to use later
            log.info("Build model....");
            BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                    .setMaxIter(iterations)
                    .theta(0.5)
                    .normalize(false)
                    .learningRate(500)
                    .useAdaGrad(false)
                    .workspaceMode(wsm)
                    .build();


            //STEP 4: establish the tsne values and save them to a file
            log.info("Store TSNE Coordinates for Plotting....");
            File outDir = testDir.newFolder();
            tsne.fit(weights);
            tsne.saveAsFile(cacheList, new File(outDir, "out.txt").getAbsolutePath());
        }
    }

    @Test
    public void testPerformance() throws Exception {

        StopWatch watch = new StopWatch();
        watch.start();
        for( int test=0; test <=1; test++){
            boolean syntheticData = test == 1;
            WorkspaceMode wsm = test == 0 ? WorkspaceMode.NONE : WorkspaceMode.ENABLED;
            log.info("Starting test: WSM={}, syntheticData={}", wsm, syntheticData);

            //STEP 1: Initialization
            int iterations = 50;
            //create an n-dimensional array of doubles
            Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);
            List<String> cacheList = new ArrayList<>(); //cacheList is a dynamic array of strings used to hold all words

            //STEP 2: Turn text input into a list of words
            INDArray weights;
            if(syntheticData){
                weights = Nd4j.rand(DataType.FLOAT, 250, 20);
            } else {
                log.info("Load & Vectorize data....");
                File wordFile = new ClassPathResource("deeplearning4j-tsne/words.txt").getFile();   //Open the file
                //Get the data of all unique word vectors
                Pair<InMemoryLookupTable, VocabCache> vectors = WordVectorSerializer.loadTxt(wordFile);
                VocabCache cache = vectors.getSecond();
                weights = vectors.getFirst().getSyn0();    //seperate weights of unique words into their own list

                for (int i = 0; i < cache.numWords(); i++)   //seperate strings of words into their own list
                    cacheList.add(cache.wordAtIndex(i));
            }

            //STEP 3: build a dual-tree tsne to use later
            log.info("Build model....");
            BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                    .setMaxIter(iterations)
                    .theta(0.5)
                    .normalize(false)
                    .learningRate(500)
                    .useAdaGrad(false)
                    .workspaceMode(wsm)
                    .build();


            //STEP 4: establish the tsne values and save them to a file
            log.info("Store TSNE Coordinates for Plotting....");
            File outDir = testDir.newFolder();
            tsne.fit(weights);
            tsne.saveAsFile(cacheList, new File(outDir, "out.txt").getAbsolutePath());
        }
        watch.stop();
        System.out.println("Elapsed time : " + watch);
    }

    @Ignore
    @Test
    public void testTSNEPerformance() throws Exception {

            for (WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}) {

                //STEP 1: Initialization
                int iterations = 50;
                //create an n-dimensional array of doubles
                Nd4j.setDataType(DataType.DOUBLE);
                List<String> cacheList = new ArrayList<>(); //cacheList is a dynamic array of strings used to hold all words

                //STEP 2: Turn text input into a list of words
                INDArray weights = Nd4j.rand(10000,300);

                StopWatch watch = new StopWatch();
                watch.start();
                //STEP 3: build a dual-tree tsne to use later
                log.info("Build model....");
                BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                        .setMaxIter(iterations)
                        .theta(0.5)
                        .normalize(false)
                        .learningRate(500)
                        .useAdaGrad(false)
                        .workspaceMode(wsm)
                        .build();

                watch.stop();
                System.out.println("Elapsed time for construction: " + watch);

                //STEP 4: establish the tsne values and save them to a file
                log.info("Store TSNE Coordinates for Plotting....");
                File outDir = testDir.newFolder();

                watch.reset();
                watch.start();
                tsne.fit(weights);
                watch.stop();
                System.out.println("Elapsed time for fit: " + watch);
                tsne.saveAsFile(cacheList, new File(outDir, "out.txt").getAbsolutePath());
            }
    }
}
