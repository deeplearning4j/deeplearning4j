package org.deeplearning4j.models.embeddings.loader;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads word 2 vec models
 * @author Adam Gibson
 */
public class WordVectorSerializer {

    private static final int MAX_SIZE = 50;
    private static final Logger log = LoggerFactory.getLogger(WordVectorSerializer.class);



    /**
     * Loads the google model
     * @param path the path to the google model
     * @return the loaded model
     * @throws IOException
     */
    public static Word2Vec loadGoogleModel(String path,boolean binary) throws IOException {
        InMemoryLookupTable lookupTable;
        VocabCache cache;
        float[] data;
        if(binary) {
          float vector;
            int words,size;
            int count = 0;
          try (BufferedInputStream bis =
                   new BufferedInputStream(path.endsWith(".gz") ?
                       new GZIPInputStream(new FileInputStream(path)) :
                       new FileInputStream(path)); DataInputStream dis = new DataInputStream(bis)) {
            words = Integer.parseInt(readString(dis));
            size = Integer.parseInt(readString(dis));
            data = new float[words * size];
            cache = new InMemoryLookupCache(false);
            lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder().cache(cache)
                .vectorLength(size).build();

            String word;
            for (int i = 0; i < words; i++) {

              word = readString(dis);
              log.info("Loading " + word + " with word " + i);
              if (word.isEmpty())
                continue;

              for (int j = 0; j < size; j++) {
                vector = dis.readFloat();
                data[count++] = vector;
              }


              cache.addWordToIndex(cache.numWords(), word);
              cache.addToken(new VocabWord(1, word));
              cache.putVocabWord(word);
            }
          }


            Word2Vec ret = new Word2Vec();

            INDArray weightMatrix = Nd4j.create(new FloatBuffer(data),new int[]{words,size});
            lookupTable.setSyn0(weightMatrix);
            ret.setVocab(cache);
            ret.setLookupTable(lookupTable);
            return ret;

        }

        else {
            BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
            String line = reader.readLine();
            String[] initial = line.split(" ");
            int words = Integer.parseInt(initial[0]);
            int layerSize = Integer.parseInt(initial[1]);
            float[] buffer = new float[words * layerSize];
            int count = 0;
            cache = new InMemoryLookupCache();



            while((line = reader.readLine()) != null) {
                String[] split = line.split(" ");
                String word = split[0];

                if(word.isEmpty())
                    continue;

                for(int i = 1; i < split.length; i++) {
                    buffer[count++] = Float.parseFloat(split[i]);

                }

                 cache.addWordToIndex(cache.numWords(),word);
                cache.addToken(new VocabWord(1,word));
                cache.putVocabWord(word);

            }


            lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder().cache(cache)
                    .vectorLength(layerSize).build();
            lookupTable.setSyn0(Nd4j.create(new FloatBuffer(buffer),new int[]{words,layerSize}));

            Word2Vec ret = new Word2Vec();
            ret.setVocab(cache);
            ret.setLookupTable(lookupTable);

            reader.close();
            return ret;

        }

    }




    private static void writeVector(float[] vec,File to) throws IOException {
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to));
        DataOutputStream dos = new DataOutputStream(bos);
        for(float v : vec) {
          dos.writeFloat(v);
        }

        bos.flush();
        bos.close();
    }


    /**
     * Read a string from a data input stream
     * Credit to: https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     * @param dis
     * @return
     * @throws IOException
     */
    private static String readString(DataInputStream dis) throws IOException {
        byte[] bytes = new byte[MAX_SIZE];
        byte b = dis.readByte();
        int i = -1;
        StringBuilder sb = new StringBuilder();
        while (b != 32 && b != 10) {
            i++;
            bytes[i] = b;
            b = dis.readByte();
            if (i == 49) {
                sb.append(new String(bytes));
                i = -1;
                bytes = new byte[MAX_SIZE];
            }



        }
        sb.append(new String(bytes, 0, i + 1));
        return sb.toString();
    }







    /**
     * Writes the word vectors to the given path.
     * Note that this assumes an in memory cache
     * @param path  the path to write
     * @throws IOException
     */
    public static void writeWordVectors(InMemoryLookupTable l,InMemoryLookupCache cache,String path) throws IOException {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path),false));
        for(int i = 0; i < l.getSyn0().rows(); i++) {
            String word = cache.wordAtIndex(i);
            if(word == null)
                continue;
            StringBuilder sb = new StringBuilder();
            sb.append(word);
            sb.append(" ");
            INDArray wordVector = l.vector(word);
            for(int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if(j < wordVector.length() - 1)
                    sb.append(" ");
            }
            sb.append("\n");
            write.write(sb.toString());

        }


        write.flush();
        write.close();


    }
    /**
     * Writes the word vectors to the given path.
     * Note that this assumes an in memory cache
     * @param vec the word2vec to write
     * @param path  the path to write
     * @throws IOException
     */
    public static void writeWordVectors(Word2Vec vec,String path) throws IOException {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path),false));
        int words = 0;
        for(String word : vec.vocab().words()) {
            if(word == null)
                continue;
            StringBuilder sb = new StringBuilder();
            sb.append(word);
            sb.append(" ");
            INDArray wordVector = vec.getWordVectorMatrix(word);
            for(int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if(j < wordVector.length() - 1)
                    sb.append(" ");
            }
            sb.append("\n");
            write.write(sb.toString());

        }


        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
        write.flush();
        write.close();


    }


    /**
     * Loads an in memory cache from the given path (sets syn0 and the vocab)
     * @param path the path of the file to load
     * @return the
     */
    public static WordVectors loadTxtVectors(File path) throws FileNotFoundException {
        Pair<WeightLookupTable,VocabCache> pair = loadTxt(path);
        WordVectorsImpl vectors = new WordVectorsImpl();
        vectors.setLookupTable(pair.getFirst());
        vectors.setVocab(pair.getSecond());
        return vectors;
    }
    /**
     * Loads an in memory cache from the given path (sets syn0 and the vocab)
     * @param path the path of the file to load
     * @return the
     */
    public static Pair<WeightLookupTable,VocabCache> loadTxt(File path) throws FileNotFoundException {
        BufferedReader write = new BufferedReader(new FileReader(path));
        VocabCache cache = new InMemoryLookupCache();

        InMemoryLookupTable l;

        LineIterator iter = IOUtils.lineIterator(write);
        List<INDArray> arrays = new ArrayList<>();
        while(iter.hasNext()) {
            String line = iter.nextLine();
            String[] split = line.split(" ");
            String word = split[0];
            VocabWord word1 = new VocabWord(1.0,word);
            cache.addToken(word1);
            cache.addWordToIndex(cache.numWords(),word);
            cache.putVocabWord(word);
            INDArray row = Nd4j.create(Nd4j.createBuffer(split.length - 1));
            for(int i = 1; i < split.length; i++)
                row.putScalar(i - 1,Float.parseFloat(split[i]));
            arrays.add(row);
        }

        INDArray syn = Nd4j.create(new int[]{arrays.size(),arrays.get(0).columns()});
        for(int i = 0; i < syn.rows(); i++)
            syn.putRow(i,arrays.get(i));

       l =  (InMemoryLookupTable) new InMemoryLookupTable.Builder().vectorLength(arrays.get(0).columns())
                .useAdaGrad(false).cache(cache)
                .build();
        Nd4j.clearNans(syn);
        l.setSyn0(syn);

        iter.close();

        return new Pair<>((WeightLookupTable) l,cache);
    }

    /**
     * Write the tsne format
     * @param vec the word vectors to use for labeling
     * @param tsne the tsne array to write
     * @param csv the file to use
     * @throws Exception
     */
    public static void writeTsneFormat(Glove vec,INDArray tsne,File csv) throws Exception {
        BufferedWriter write = new BufferedWriter(new FileWriter(csv));
        int words = 0;
        InMemoryLookupCache l = (InMemoryLookupCache) vec.vocab();
        for(String word : vec.vocab().words()) {
            if(word == null)
                continue;
            StringBuilder sb = new StringBuilder();
            INDArray wordVector = tsne.getRow(l.wordFor(word).getIndex());
            for(int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if(j < wordVector.length() - 1)
                    sb.append(",");
            }
            sb.append(",");
            sb.append(word);
            sb.append(" ");

            sb.append("\n");
            write.write(sb.toString());

        }

        log.info("Wrote " + words + " with size of " + vec.lookupTable().getVectorLength());
        write.flush();
        write.close();


    }
    /**
     * Write the tsne format
     * @param vec the word vectors to use for labeling
     * @param tsne the tsne array to write
     * @param csv the file to use
     * @throws Exception
     */
    public static void writeTsneFormat(Word2Vec vec,INDArray tsne,File csv) throws Exception {
        BufferedWriter write = new BufferedWriter(new FileWriter(csv));
        int words = 0;
        InMemoryLookupCache l = (InMemoryLookupCache) vec.vocab();
        for(String word : vec.vocab().words()) {
            if(word == null)
                continue;
            StringBuilder sb = new StringBuilder();
            INDArray wordVector = tsne.getRow(l.wordFor(word).getIndex());
            for(int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if(j < wordVector.length() - 1)
                    sb.append(",");
            }
            sb.append(",");
            sb.append(word);
            sb.append(" ");

            sb.append("\n");
            write.write(sb.toString());

        }

        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
        write.flush();
        write.close();


    }



}
