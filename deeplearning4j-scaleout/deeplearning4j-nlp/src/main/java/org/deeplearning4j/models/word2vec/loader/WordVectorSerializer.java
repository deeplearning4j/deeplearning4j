package org.deeplearning4j.models.word2vec.loader;

import java.io.*;
import java.util.*;

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
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads word 2 vec models
 * @author Adam Gibson
 */
public class WordVectorSerializer {

    private static final int MAX_SIZE = 50;
    private static Logger log = LoggerFactory.getLogger(WordVectorSerializer.class);



    /**
     * Loads the google model
     * @param path the path to the google model
     * @return the loaded model
     * @throws IOException
     */
    public static Word2Vec loadGoogleModel(String path,boolean binary) throws IOException {
        WeightLookupTable lookupTable;
        VocabCache cache;
        List<File> vectorPaths = new ArrayList<>();

        File rootDir = new File("." + UUID.randomUUID().toString());
        if(!rootDir.mkdirs())
            throw new IllegalStateException("Unable to create directory for word vectors");

        if(binary) {
            DataInputStream dis = null;
            BufferedInputStream bis = null;
            double len;
            float vector;
            int words,size = 0;

            try {
                bis = new BufferedInputStream(new FileInputStream(path));
                dis = new DataInputStream(bis);
                words = Integer.parseInt(readString(dis));
                size = Integer.parseInt(readString(dis));

                cache = new InMemoryLookupCache();
                lookupTable = new InMemoryLookupTable.Builder().cache(cache)
                        .vectorLength(size).build();

                String word;
                float[] vectors;
                for (int i = 0; i < words; i++) {

                    word = readString(dis);

                    if(word.isEmpty())
                        continue;


                    vectors = new float[size];
                    len = 0;
                    for (int j = 0; j < size; j++) {
                        vector = readFloat(dis);
                        len += vector * vector;
                        vectors[j] = vector;
                    }
                    len = Math.sqrt(len);

                    for (int j = 0; j < size; j++)
                        vectors[j] /= len;

                    File write = new File(rootDir,String.valueOf(i));
                    vectorPaths.add(write);

                    writeVector(vectors,write);

                    cache.addWordToIndex(cache.numWords(),word);
                    cache.addToken(new VocabWord(1,word));
                    cache.putVocabWord(word);
                    dis.read();
                }
            }
            finally {
                bis.close();
                dis.close();
            }


            Word2Vec ret = new Word2Vec();
            lookupTable.resetWeights();

            for(int i = 0; i < vectorPaths.size(); i++) {
                float[] read = readVec(vectorPaths.get(i),size);
                lookupTable.putVector(cache.wordAtIndex(i),Nd4j.create(read));
                vectorPaths.get(i).delete();
            }

            ret.setVocab(cache);
            ret.setLookupTable(lookupTable);
            ret.setLayerSize(size);
            rootDir.delete();
            return ret;

        }

        else {
            BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
            String line = reader.readLine();
            String[] initial = line.split(" ");
            int words = Integer.parseInt(initial[0]);
            int layerSize = Integer.parseInt(initial[1]);
            cache = new InMemoryLookupCache();



            while((line = reader.readLine()) != null) {
                String[] split = line.split(" ");
                String word = split[0];

                if(word.isEmpty())
                    continue;

                float[] buffer = new float[layerSize];
                for(int i = 1; i < split.length; i++) {
                    buffer[i - 1] = Float.parseFloat(split[i]);
                }

                File vecFile = new File(rootDir,String.valueOf(cache.numWords()));
                writeVector(buffer,vecFile);
                vectorPaths.add(vecFile);
                cache.addWordToIndex(cache.numWords(),word);
                cache.addToken(new VocabWord(1,word));
                cache.putVocabWord(word);

            }


            lookupTable = new InMemoryLookupTable.Builder().cache(cache)
                    .vectorLength(layerSize).build();


            lookupTable.resetWeights();

            for(int i = 0; i < words; i++) {
                float[] read = readVec(vectorPaths.get(i),layerSize);
                lookupTable.putVector(cache.wordAtIndex(i),Nd4j.create(read));
                vectorPaths.get(i).delete();
            }

            Word2Vec ret = new Word2Vec();
            ret.setVocab(cache);
            ret.setLookupTable(lookupTable);
            ret.setLayerSize(layerSize);

            reader.close();
            rootDir.delete();
            return ret;

        }

    }



    private static float[] readVec(File from,int length) throws IOException {
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream(from));
        DataInputStream dis = new DataInputStream(bis);
        float[] ret = new float[length];
        for(int i = 0; i < length; i++) {
            ret[i] = dis.readFloat();
        }

        dis.close();

        return ret;


    }

    private static void writeVector(float[] vec,File to) throws IOException {
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(to));
        DataOutputStream dos = new DataOutputStream(bos);
        for(int i = 0; i < vec.length; i++) {
            dos.writeFloat(vec[i]);
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
     * Credit to: https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     * @param is
     * @return
     * @throws IOException
     */
    public static float readFloat(InputStream is) throws IOException {
        byte[] bytes = new byte[4];
        is.read(bytes);
        return getFloat(bytes);
    }

    /**
     * Read float from byte array, credit to:
     * Credit to: https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     *
     * @param b
     * @return
     */
    public static float getFloat(byte[] b) {
        int accum = 0;
        accum = accum | (b[0] & 0xff) << 0;
        accum = accum | (b[1] & 0xff) << 8;
        accum = accum | (b[2] & 0xff) << 16;
        accum = accum | (b[3] & 0xff) << 24;
        return Float.intBitsToFloat(accum);
    }


    /**
     * Writes the word vectors to the given path.
     * Note that this assumes an in memory cache
     * @param path  the path to write
     * @throws IOException
     */
    public static void writeWordVectors(InMemoryLookupTable l,InMemoryLookupCache cache,String path) throws IOException {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path),false));
        int words = 0;
        for(int i = 0; i < l.getSyn0().rows(); i++) {
            String word = cache.wordAtIndex(i);
            if(word == null)
                continue;
            StringBuffer sb = new StringBuffer();
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
            StringBuffer sb = new StringBuffer();
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
        vectors.setLayerSize(pair.getFirst().layerSize());
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

        InMemoryLookupTable l = (InMemoryLookupTable) new InMemoryLookupTable.Builder().vectorLength(100)
                .useAdaGrad(false).cache(cache)
                .build();

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
            StringBuffer sb = new StringBuffer();
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
            StringBuffer sb = new StringBuffer();
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
