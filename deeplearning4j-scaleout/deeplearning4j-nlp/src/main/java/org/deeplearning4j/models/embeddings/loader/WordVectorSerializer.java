/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.embeddings.loader;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Loads word 2 vec models
 *
 * @author Adam Gibson
 */
public class WordVectorSerializer
{
    private static final boolean DEFAULT_LINEBREAKS = false;
    private static final int MAX_SIZE = 50;
    private static final Logger log = LoggerFactory.getLogger(WordVectorSerializer.class);

    /**
     * Loads the google model
     *
     * @param modelFile
     *            the path to the google model
     * @param binary
     *            read from binary file format (if set to true) or from text file format.
     * @return the loaded model
     * @throws IOException
     */
    public static Word2Vec loadGoogleModel(File modelFile, boolean binary)
        throws IOException
    {
        return loadGoogleModel(modelFile, binary, DEFAULT_LINEBREAKS);
    }

    /**
     * Loads the Google model.
     * 
     * @param modelFile
     *            the input file
     * @param binary
     *            read from binary or text file format
     * @param lineBreaks
     *            if true, the input file is expected to terminate each line with a line break. This
     *            is typically the case for files created with recent versions of Word2Vec, but not
     *            for the downloadable model files.
     * @return a {@link Word2Vec} object
     * @throws IOException
     * @author Carsten Schnober
     */
    public static Word2Vec loadGoogleModel(File modelFile, boolean binary, boolean lineBreaks)
        throws IOException
    {
        return binary ? readBinaryModel(modelFile, lineBreaks) : readTextModel(modelFile);
    }

    /**
     * @param modelFile
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     * @throws NumberFormatException
     */
    private static Word2Vec readTextModel(File modelFile)
        throws IOException, NumberFormatException
    {
        InMemoryLookupTable lookupTable;
        VocabCache cache;
        INDArray syn0;
        BufferedReader reader = new BufferedReader(new FileReader(modelFile));
        String line = reader.readLine();
        String[] initial = line.split(" ");
        int words = Integer.parseInt(initial[0]);
        int layerSize = Integer.parseInt(initial[1]);
        syn0 = Nd4j.create(words, layerSize);

        cache = new InMemoryLookupCache();

        int currLine = 0;
        while ((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            assert split.length == layerSize + 1;
            String word = split[0];

            float[] vector = new float[split.length - 1];
            for (int i = 1; i < split.length; i++) {
                vector[i - 1] = Float.parseFloat(split[i]);
            }

            syn0.putRow(currLine, Transforms.unitVec(Nd4j.create(vector)));

            cache.addWordToIndex(cache.numWords(), word);
            cache.addToken(new VocabWord(1, word));
            cache.putVocabWord(word);

            currLine++;
        }

        lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder().cache(cache)
                .vectorLength(layerSize).build();
        lookupTable.setSyn0(syn0);

        Word2Vec ret = new Word2Vec();
        ret.setVocab(cache);
        ret.setLookupTable(lookupTable);

        reader.close();
        return ret;
    }

    /**
     * Read a binary word2vec file.
     * 
     * @param modelFile
     *            the File to read
     * @param linebreaks
     *            if true, the reader expects each word/vector to be in a separate line, terminated
     *            by a line break
     * @return a {@link Word2Vec model}
     * @throws NumberFormatException
     * @throws IOException
     * @throws FileNotFoundException
     */
    private static Word2Vec readBinaryModel(File modelFile, boolean linebreaks)
        throws NumberFormatException, IOException
    {
        InMemoryLookupTable lookupTable;
        VocabCache cache;
        INDArray syn0;
        int words, size;
        try (BufferedInputStream bis = new BufferedInputStream(
                GzipUtils.isCompressedFilename(modelFile.getName())
                        ? new GZIPInputStream(new FileInputStream(modelFile))
                        : new FileInputStream(modelFile));
                DataInputStream dis = new DataInputStream(bis)) {
            words = Integer.parseInt(readString(dis));
            size = Integer.parseInt(readString(dis));
            syn0 = Nd4j.create(words, size);
            cache = new InMemoryLookupCache(false);
            lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder().cache(cache)
                    .vectorLength(size).build();

            String word;
            for (int i = 0; i < words; i++) {

                word = readString(dis);
                log.trace("Loading " + word + " with word " + i);

                float[] vector = new float[size];

                for (int j = 0; j < size; j++) {
                    vector[j] = readFloat(dis);
                }

                syn0.putRow(i, Transforms.unitVec(Nd4j.create(vector)));

                cache.addWordToIndex(cache.numWords(), word);
                cache.addToken(new VocabWord(1, word));
                cache.putVocabWord(word);

                if (linebreaks) {
                    dis.readByte(); // line break
                }
            }
        }

        Word2Vec ret = new Word2Vec();

        lookupTable.setSyn0(syn0);
        ret.setVocab(cache);
        ret.setLookupTable(lookupTable);
        return ret;

    }

    /**
     * Read a float from a data input stream Credit to:
     * https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     *
     * @param is
     * @return
     * @throws IOException
     */
    public static float readFloat(InputStream is)
        throws IOException
    {
        byte[] bytes = new byte[4];
        is.read(bytes);
        return getFloat(bytes);
    }

    /**
     * Read a string from a data input stream Credit to:
     * https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     *
     * @param b
     * @return
     * @throws IOException
     */
    public static float getFloat(byte[] b)
    {
        int accum = 0;
        accum = accum | (b[0] & 0xff) << 0;
        accum = accum | (b[1] & 0xff) << 8;
        accum = accum | (b[2] & 0xff) << 16;
        accum = accum | (b[3] & 0xff) << 24;
        return Float.intBitsToFloat(accum);
    }

    /**
     * Read a string from a data input stream Credit to:
     * https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
     *
     * @param dis
     * @return
     * @throws IOException
     */
    public static String readString(DataInputStream dis)
        throws IOException
    {
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
     * Writes the word vectors to the given path. Note that this assumes an in memory cache
     * 
     * @param lookupTable
     * @param cache
     *
     * @param path
     *            the path to write
     * @throws IOException
     */
    public static void writeWordVectors(InMemoryLookupTable lookupTable, InMemoryLookupCache cache,
            String path)
                throws IOException
    {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path), false));
        for (int i = 0; i < lookupTable.getSyn0().rows(); i++) {
            String word = cache.wordAtIndex(i);
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(word.replaceAll(" ", "_"));
            sb.append(" ");
            INDArray wordVector = lookupTable.vector(word);
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(" ");
                }
            }
            sb.append("\n");
            write.write(sb.toString());

        }

        write.flush();
        write.close();

    }

    /**
     * Writes the word vectors to the given path. Note that this assumes an in memory cache
     *
     * @param vec
     *            the word2vec to write
     * @param path
     *            the path to write
     * @throws IOException
     */
    public static void writeWordVectors(Word2Vec vec, String path)
        throws IOException
    {
        BufferedWriter write = new BufferedWriter(new FileWriter(new File(path), false));
        int words = 0;
        for (String word : vec.vocab().words()) {
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(word.replaceAll(" ", "_"));
            sb.append(" ");
            INDArray wordVector = vec.getWordVectorMatrix(word);
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(" ");
                }
            }
            sb.append("\n");
            write.write(sb.toString());
            words++;

        }

        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
        write.flush();
        write.close();

    }

    /**
     * Load word vectors for the given vocab and table
     *
     * @param table
     *            the weights to use
     * @param vocab
     *            the vocab to use
     * @return wordvectors based on the given parameters
     */
    public static WordVectors fromTableAndVocab(WeightLookupTable table, VocabCache vocab)
    {
        WordVectorsImpl vectors = new WordVectorsImpl();
        vectors.setLookupTable(table);
        vectors.setVocab(vocab);
        return vectors;
    }

    /**
     * Load word vectors from the given pair
     *
     * @param pair
     *            the given pair
     * @return a read only word vectors impl based on the given lookup table and vocab
     */
    public static WordVectors fromPair(Pair<InMemoryLookupTable, VocabCache> pair)
    {
        WordVectorsImpl vectors = new WordVectorsImpl();
        vectors.setLookupTable(pair.getFirst());
        vectors.setVocab(pair.getSecond());
        return vectors;
    }

    /**
     * Loads an in memory cache from the given path (sets syn0 and the vocab)
     *
     * @param vectorsFile
     *            the path of the file to load\
     * @return
     * @throws FileNotFoundException
     *             if the file does not exist
     */
    public static WordVectors loadTxtVectors(File vectorsFile)
        throws FileNotFoundException
    {
        Pair<InMemoryLookupTable, VocabCache> pair = loadTxt(vectorsFile);
        return fromPair(pair);
    }

    /**
     * Loads an in memory cache from the given path (sets syn0 and the vocab)
     *
     * @param vectorsFile
     *            the path of the file to load
     * @return
     * @throws FileNotFoundException
     */
    public static Pair<InMemoryLookupTable, VocabCache> loadTxt(File vectorsFile)
        throws FileNotFoundException
    {
        BufferedReader write = new BufferedReader(new FileReader(vectorsFile));
        VocabCache cache = new InMemoryLookupCache();

        InMemoryLookupTable lookupTable;

        LineIterator iter = IOUtils.lineIterator(write);
        List<INDArray> arrays = new ArrayList<>();
        while (iter.hasNext()) {
            String line = iter.nextLine();
            String[] split = line.split(" ");
            String word = split[0];
            VocabWord word1 = new VocabWord(1.0, word);
            cache.addToken(word1);
            cache.addWordToIndex(cache.numWords(), word);
            word1.setIndex(cache.numWords());
            cache.putVocabWord(word);
            INDArray row = Nd4j.create(Nd4j.createBuffer(split.length - 1));
            for (int i = 1; i < split.length; i++) {
                row.putScalar(i - 1, Float.parseFloat(split[i]));
            }
            arrays.add(row);
        }

        INDArray syn = Nd4j.create(new int[] { arrays.size(), arrays.get(0).columns() });
        for (int i = 0; i < syn.rows(); i++) {
            syn.putRow(i, arrays.get(i));
        }

        lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
                .vectorLength(arrays.get(0).columns())
                .useAdaGrad(false).cache(cache)
                .build();
        Nd4j.clearNans(syn);
        lookupTable.setSyn0(syn);

        iter.close();

        return new Pair<>(lookupTable, cache);
    }

    /**
     * Write the tsne format
     *
     * @param vec
     *            the word vectors to use for labeling
     * @param tsne
     *            the tsne array to write
     * @param csv
     *            the file to use
     * @throws Exception
     */
    public static void writeTsneFormat(Glove vec, INDArray tsne, File csv)
        throws Exception
    {
        BufferedWriter write = new BufferedWriter(new FileWriter(csv));
        int words = 0;
        InMemoryLookupCache l = (InMemoryLookupCache) vec.vocab();
        for (String word : vec.vocab().words()) {
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            INDArray wordVector = tsne.getRow(l.wordFor(word).getIndex());
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(",");
                }
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
     *
     * @param vec
     *            the word vectors to use for labeling
     * @param tsne
     *            the tsne array to write
     * @param csv
     *            the file to use
     * @throws Exception
     */
    public static void writeTsneFormat(Word2Vec vec, INDArray tsne, File csv)
        throws Exception
    {
        BufferedWriter write = new BufferedWriter(new FileWriter(csv));
        int words = 0;
        InMemoryLookupCache l = (InMemoryLookupCache) vec.vocab();
        for (String word : vec.vocab().words()) {
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            INDArray wordVector = tsne.getRow(l.wordFor(word).getIndex());
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(",");
                }
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
