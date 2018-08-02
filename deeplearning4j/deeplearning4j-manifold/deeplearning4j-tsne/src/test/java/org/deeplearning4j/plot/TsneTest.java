//package org.deeplearning4j.plot;
//
//import lombok.extern.slf4j.Slf4j;
//import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
//import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
//import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
//import org.deeplearning4j.nn.conf.WorkspaceMode;
//import org.junit.Test;
//import org.nd4j.linalg.api.buffer.DataBuffer;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.io.ClassPathResource;
//import org.nd4j.linalg.primitives.Pair;
//
//import java.io.File;
//import java.util.ArrayList;
//import java.util.List;
//
//@Slf4j
//public class TsneTest {
//
//    @Test
//    public void testSimple() throws Exception {
//        //Simple sanity check
//
//        for(WorkspaceMode wsm : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}){
//
//            //STEP 1: Initialization
//            int iterations = 100;
//            //create an n-dimensional array of doubles
//            Nd4j.setDataType(DataBuffer.Type.DOUBLE);
//            List<String> cacheList = new ArrayList<>(); //cacheList is a dynamic array of strings used to hold all words
//
//            //STEP 2: Turn text input into a list of words
//            log.info("Load & Vectorize data....");
//            File wordFile = new ClassPathResource("deeplearning4j-tsne/words.txt").getFile();   //Open the file
//            //Get the data of all unique word vectors
//            Pair<InMemoryLookupTable,VocabCache> vectors = WordVectorSerializer.loadTxt(wordFile);
//            VocabCache cache = vectors.getSecond();
//            INDArray weights = vectors.getFirst().getSyn0();    //seperate weights of unique words into their own list
//
//            for(int i = 0; i < cache.numWords(); i++)   //seperate strings of words into their own list
//                cacheList.add(cache.wordAtIndex(i));
//
//            //STEP 3: build a dual-tree tsne to use later
//            log.info("Build model....");
//            BarnesHutTsne tsne = new BarnesHutTsne.Builder()
//                    .setMaxIter(iterations).theta(0.5)
//                    .normalize(false)
//                    .learningRate(500)
//                    .useAdaGrad(false)
//                    .workspaceMode(wsm)
//                    .build();
//
//            //STEP 4: establish the tsne values and save them to a file
//            log.info("Store TSNE Coordinates for Plotting....");
//            String outputFile = "target/archive-tmp/tsne-standard-coords.csv";
//            (new File(outputFile)).getParentFile().mkdirs();
//
//            tsne.fit(weights);
//            tsne.saveAsFile(cacheList, outputFile);
//
//
//        }
//    }
//
//}
