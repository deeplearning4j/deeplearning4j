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

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MapperFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.apache.commons.codec.binary.Base64;
import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceElementFactory;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.glove.Glove;
import org.deeplearning4j.models.sequencevectors.serialization.VocabWordFactory;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;

/**
 * Loads word 2 vec models
 *
 * @author Adam Gibson
 */
public class WordVectorSerializer {
    private static final boolean DEFAULT_LINEBREAKS = false;
    private static final boolean HAS_HEADER = true;
    private static final int MAX_SIZE = 50;
    private static final String whitespaceReplacement = "_Az92_";
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
    public static WordVectors loadGoogleModel(File modelFile, boolean binary)
            throws IOException {
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
    public static WordVectors loadGoogleModel(File modelFile, boolean binary, boolean lineBreaks)
            throws IOException {
        return binary ? readBinaryModel(modelFile, lineBreaks, true) : WordVectorSerializer.fromPair(loadTxt(modelFile));
    }

    /**
     *
     * Loads the Google model without normalization being applied.
     *
     * PLEASE NOTE: Use this method only if you understand why you need not-normalized model. In all other cases please use loadGoogleModel() instead.
     *
     * @param modelFile
     * @param binary
     * @param lineBreaks
     * @return
     * @throws IOException
     */
    public static WordVectors loadGoogleModelNonNormalized(File modelFile, boolean binary, boolean lineBreaks) throws IOException {
        return binary ? readBinaryModel(modelFile, lineBreaks, false) : WordVectorSerializer.fromPair(loadTxt(modelFile));
    }

    /**
     * @param modelFile
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     * @throws NumberFormatException
     */
    private static Word2Vec readTextModel(File modelFile)
            throws IOException, NumberFormatException {
        InMemoryLookupTable lookupTable;
        VocabCache cache;
        INDArray syn0;
        Word2Vec ret = new Word2Vec();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                GzipUtils.isCompressedFilename(modelFile.getName())
                        ? new GZIPInputStream(new FileInputStream(modelFile))
                        : new FileInputStream(modelFile), "UTF-8"))) {
            String line = reader.readLine();
            String[] initial = line.split(" ");
            int words = Integer.parseInt(initial[0]);
            int layerSize = Integer.parseInt(initial[1]);
            syn0 = Nd4j.create(words, layerSize);

            cache = new InMemoryLookupCache(false);

            int currLine = 0;
            while ((line = reader.readLine()) != null) {
                String[] split = line.split(" ");
                assert split.length == layerSize + 1;
                String word = split[0].replaceAll(whitespaceReplacement, " ");

                float[] vector = new float[split.length - 1];
                for (int i = 1; i < split.length; i++) {
                    vector[i - 1] = Float.parseFloat(split[i]);
                }

                syn0.putRow(currLine, Nd4j.create(vector));

                cache.addWordToIndex(cache.numWords(), word);
                cache.addToken(new VocabWord(1, word));
                cache.putVocabWord(word);

                currLine++;
            }

            lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder().
                    cache(cache).vectorLength(layerSize).build();
            lookupTable.setSyn0(syn0);

            ret.setVocab(cache);
            ret.setLookupTable(lookupTable);
        }
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
    private static Word2Vec readBinaryModel(File modelFile, boolean linebreaks, boolean normalize)
            throws NumberFormatException, IOException
    {
        InMemoryLookupTable<VocabWord> lookupTable;
        VocabCache<VocabWord> cache;
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
            lookupTable = (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                    .cache(cache)
                    .vectorLength(size).build();

            String word;
            for (int i = 0; i < words; i++) {

                word = readString(dis);
                log.trace("Loading " + word + " with word " + i);

                float[] vector = new float[size];

                for (int j = 0; j < size; j++) {
                    vector[j] = readFloat(dis);
                }


                syn0.putRow(i, normalize ? Transforms.unitVec(Nd4j.create(vector)) : Nd4j.create(vector));

                cache.addToken(new VocabWord(1, word));
                cache.addWordToIndex(cache.numWords(), word);
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
                sb.append(new String(bytes, "UTF-8"));
                i = -1;
                bytes = new byte[MAX_SIZE];
            }
        }
        sb.append(new String(bytes, 0, i + 1, "UTF-8"));
        return sb.toString();
    }

    /**
     * This mehod writes word vectors to the given path.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param path
     * @param <T>
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable, String path) throws IOException {
        try {
            writeWordVectors(lookupTable, new FileOutputStream(path));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This mehod writes word vectors to the given file.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param file
     * @param <T>
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable, File file) throws IOException {
        try {
            writeWordVectors(lookupTable, new FileOutputStream(file));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This mehod writes word vectors to the given OutputStream.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param stream
     * @param <T>
     * @throws IOException
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable, OutputStream stream) throws IOException {
        VocabCache<T> vocabCache = lookupTable.getVocabCache();

        PrintWriter writer = new PrintWriter(new OutputStreamWriter(stream, "UTF-8"));

        for (int x = 0; x < vocabCache.numWords(); x++) {
            T element = vocabCache.elementAtIndex(x);

            StringBuilder builder = new StringBuilder();

            builder.append(element.getLabel().replaceAll(" ", whitespaceReplacement)).append(" ");
            INDArray vec = lookupTable.vector(element.getLabel());
            for (int i = 0; i < vec.length(); i++) {
                builder.append(vec.getDouble(i));
                if (i < vec.length() - 1) builder.append(" ");
            }
            writer.println(builder.toString());
        }
        writer.flush();
        writer.close();
    }


    /**
     * This method saves paragraph vectors to the given file.
     *
     * @param vectors
     * @param path
     */
    public static void writeWordVectors(@NonNull ParagraphVectors vectors, @NonNull File path) {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * This method saves paragraph vectors to the given path.
     *
     * @param vectors
     * @param path
     */
    public static void writeWordVectors(@NonNull ParagraphVectors vectors, @NonNull String path) {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Restores previously serialized ParagraphVectors model
     *
     * @param path Path to file that contains previously serialized model
     * @return
     */
    public static ParagraphVectors readParagraphVectorsFromText(@NonNull String path) {
        try {
            return readParagraphVectorsFromText(new FileInputStream(path));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Restores previously serialized ParagraphVectors model
     *
     * @param file File that contains previously serialized model
     * @return
     */
    public static ParagraphVectors readParagraphVectorsFromText(@NonNull File file) {
        try {
            return readParagraphVectorsFromText(new FileInputStream(file));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Restores previously serialized ParagraphVectors model
     *
     * @param stream InputStream that contains previously serialized model
     * @return
     */
    public static ParagraphVectors readParagraphVectorsFromText(@NonNull InputStream stream) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream, "UTF-8"));
            ArrayList<String> labels = new ArrayList<>();
            ArrayList<INDArray> arrays = new ArrayList<>();
            VocabCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();
            String line = "";
            while ((line = reader.readLine()) != null) {
                String[] split = line.split(" ");
                split[1] = split[1].replaceAll(whitespaceReplacement, " ");
                VocabWord word = new VocabWord(1.0, split[1]);
                if (split[0].equals("L")) {
                    // we have label element here
                    word.setSpecial(true);
                    word.markAsLabel(true);
                    labels.add(word.getLabel());
                } else if (split[0].equals("E")) {
                    // we have usual element, aka word here
                    word.setSpecial(false);
                    word.markAsLabel(false);
                } else throw new IllegalStateException("Source stream doesn't looks like ParagraphVectors serialized model");

                // this particular line is just for backward compatibility with InMemoryLookupCache
                word.setIndex(vocabCache.numWords());

                vocabCache.addToken(word);
                vocabCache.addWordToIndex(word.getIndex(), word.getLabel());

                // backward compatibility code
                vocabCache.putVocabWord(word.getLabel());

                INDArray row = Nd4j.create(1, split.length - 2);
                for (int i = 2; i < split.length; i++) {
                    row.putScalar(i - 2, Float.parseFloat(split[i]));
                }

                arrays.add(row);
            }

            // now we create syn0 matrix, using previously fetched rows
            INDArray syn = Nd4j.create(new int[]{arrays.size(), arrays.get(0).columns()});
            for (int i = 0; i < syn.rows(); i++) {
                syn.putRow(i, arrays.get(i));
            }


            InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                    .vectorLength(arrays.get(0).columns())
                    .useAdaGrad(false)
                    .cache(vocabCache)
                    .build();
            Nd4j.clearNans(syn);
            lookupTable.setSyn0(syn);

            LabelsSource source = new LabelsSource(labels);
            ParagraphVectors vectors = new ParagraphVectors.Builder()
                    .labelsSource(source)
                    .vocabCache(vocabCache)
                    .lookupTable(lookupTable)
                    .modelUtils(new BasicModelUtils<VocabWord>())
                    .build();

            try {
                reader.close();
            } catch (Exception e) {
                ;
            }

            return vectors;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves GloVe model to the given output stream.
     *
     * @param vectors GloVe model to be saved
     * @param file path where model should be saved to
     */
    public static void writeWordVectors(@NonNull Glove vectors, @NonNull File file) {
        try (FileOutputStream fos = new FileOutputStream(file)) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves GloVe model to the given output stream.
     *
     * @param vectors GloVe model to be saved
     * @param path path where model should be saved to
     */
    public static void writeWordVectors(@NonNull Glove vectors, @NonNull String path) {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves GloVe model to the given OutputStream
     *
     * @param vectors GloVe model to be saved
     * @param stream OutputStream where model should be saved to
     */
    public static void writeWordVectors(@NonNull Glove vectors, @NonNull OutputStream stream) {
        try {
            writeWordVectors(vectors.lookupTable(), stream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves paragraph vectors to the given output stream.
     *
     * @param vectors
     * @param stream
     */
    public static void writeWordVectors(ParagraphVectors vectors, OutputStream stream) {

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream, "UTF-8"))) {
        /*
            This method acts similary to w2v csv serialization, except of additional tag for labels
         */

            VocabCache<VocabWord> vocabCache = vectors.getVocab();
            for (VocabWord word : vocabCache.vocabWords()) {
                StringBuilder builder = new StringBuilder();

                builder.append(word.isLabel() ? "L" : "E").append(" ");
                builder.append(word.getLabel().replaceAll(" ", whitespaceReplacement)).append(" ");

                INDArray vector = vectors.getWordVectorMatrix(word.getLabel());
                for (int j = 0; j < vector.length(); j++) {
                    builder.append(vector.getDouble(j));
                    if (j < vector.length() - 1) {
                        builder.append(" ");
                    }
                }

                writer.write(builder.append("\n").toString());
            }

            writer.flush();
            writer.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
            throws IOException {
        BufferedWriter write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(path), false), "UTF-8"));
        for (int i = 0; i < lookupTable.getSyn0().rows(); i++) {
            String word = cache.wordAtIndex(i);
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(word.replaceAll(" ", whitespaceReplacement));
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

    private static ObjectMapper getModelMapper() {
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        return ret;
    }

    /**
     * Saves full Word2Vec model in the way, that allows model updates without being rebuilt from scratches
     *
     * @param vec - The Word2Vec instance to be saved
     * @param path - the path for json to be saved
     */
    public static void writeFullModel(@NonNull Word2Vec vec, @NonNull String path) {
        /*
            Basically we need to save:
                    1. WeightLookupTable, especially syn0 and syn1 matrices
                    2. VocabCache, including only WordCounts
                    3. Settings from Word2Vect model: workers, layers, etc.
         */

        PrintWriter printWriter = null;
        try {
            printWriter = new PrintWriter(new OutputStreamWriter(new FileOutputStream(path), "UTF-8"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        WeightLookupTable<VocabWord> lookupTable = vec.getLookupTable();
        VocabCache<VocabWord> vocabCache = vec.getVocab(); // ((InMemoryLookupTable) lookupTable).getVocab(); //vec.getVocab();


        if (!(lookupTable instanceof InMemoryLookupTable)) throw new IllegalStateException("At this moment only InMemoryLookupTable is supported.");

        VectorsConfiguration conf = vec.getConfiguration();
        conf.setVocabSize(vocabCache.numWords());


        printWriter.println(conf.toJson());
        log.info("Word2Vec conf. JSON: " + conf.toJson());
        /*
            We have the following map:
            Line 0 - VectorsConfiguration JSON string
            Line 1 - expTable
            Line 2 - table

            All following lines are vocab/weight lookup table saved line by line as VocabularyWord JSON representation
         */

        // actually we don't need expTable, since it produces exact results on subsequent runs untill you dont modify expTable size :)
        // saving ExpTable just for "special case in future"
        StringBuilder builder = new StringBuilder();
        for (int x = 0; x < ((InMemoryLookupTable) lookupTable).getExpTable().length; x++) {
            builder.append(((InMemoryLookupTable) lookupTable).getExpTable()[x]).append(" ");
        }
        printWriter.println(builder.toString().trim());

        // saving table, available only if negative sampling is used
        if (conf.getNegative() > 0 && ((InMemoryLookupTable) lookupTable).getTable() != null) {
            builder = new StringBuilder();
            for (int x = 0; x < ((InMemoryLookupTable) lookupTable).getTable().columns(); x++) {
                builder.append(((InMemoryLookupTable) lookupTable).getTable().getDouble(x)).append(" ");
            }
            printWriter.println(builder.toString().trim());
        } else printWriter.println("");



        List<VocabWord> words = new ArrayList<VocabWord>(vocabCache.vocabWords());
        for (SequenceElement word: words) {
            VocabularyWord vw = new VocabularyWord(word.getLabel());
            vw.setCount(vocabCache.wordFrequency(word.getLabel()));

            vw.setHuffmanNode(VocabularyHolder.buildNode(word.getCodes(), word.getPoints(), word.getCodeLength(), word.getIndex()));


            // writing down syn0
            INDArray syn0 = ((InMemoryLookupTable) lookupTable).getSyn0().getRow(vocabCache.indexOf(word.getLabel()));
            double[] dsyn0 = new double[syn0.columns()];
            for (int x =0; x < conf.getLayersSize(); x++) {
                dsyn0[x] = syn0.getDouble(x);
            }
            vw.setSyn0(dsyn0);

            // writing down syn1
            INDArray syn1 = ((InMemoryLookupTable) lookupTable).getSyn1().getRow(vocabCache.indexOf(word.getLabel()));
            double[] dsyn1 = new double[syn1.columns()];
            for (int x =0; x < syn1.columns(); x++) {
                dsyn1[x] = syn1.getDouble(x);
            }
            vw.setSyn1(dsyn1);

            // writing down syn1Neg, if negative sampling is used
            if (conf.getNegative() > 0 && ((InMemoryLookupTable) lookupTable).getSyn1Neg() != null) {
                INDArray syn1Neg = ((InMemoryLookupTable) lookupTable).getSyn1Neg().getRow(vocabCache.indexOf(word.getLabel()));
                double[] dsyn1Neg = new double[syn1Neg.columns()];
                for (int x = 0; x < syn1Neg.columns(); x++) {
                    dsyn1Neg[x] = syn1Neg.getDouble(x);
                }
                vw.setSyn1Neg(dsyn1Neg);
            }


            // in case of UseAdaGrad == true - we should save gradients for each word in vocab
            if (conf.isUseAdaGrad() && ((InMemoryLookupTable) lookupTable).isUseAdaGrad()) {
                INDArray gradient = word.getHistoricalGradient();
                if (gradient == null) gradient = Nd4j.zeros(word.getCodes().size());
                double ada[] = new double[gradient.columns()];
                for (int x = 0; x < gradient.columns(); x++) {
                    ada[x] = gradient.getDouble(x);
                }
                vw.setHistoricalGradient(ada);
            }

            printWriter.println(vw.toJson());
        }

        // at this moment we have whole vocab serialized
        printWriter.flush();
        printWriter.close();
    }

    /**
     *
     * @param path - path to previously stored w2v json model
     * @return - Word2Vec instance
     */
    public static Word2Vec loadFullModel(@NonNull String path) throws FileNotFoundException {
        /*
            // TODO: implementation is in process
            We need to restore:
                     1. WeightLookupTable, including syn0 and syn1 matrices
                     2. VocabCache + mark it as SPECIAL, to avoid accidental word removals
         */
        BasicLineIterator iterator = new BasicLineIterator(new File(path));

        // first 3 lines should be processed separately
        String confJson  = iterator.nextSentence();
        log.info("Word2Vec conf. JSON: " + confJson);
        VectorsConfiguration configuration = VectorsConfiguration.fromJson(confJson);


        // actually we dont need expTable, since it produces exact results on subsequent runs untill you dont modify expTable size :)
        String eTable = iterator.nextSentence();
        double[] expTable;


        String nTable = iterator.nextSentence();
        if (configuration.getNegative() > 0) {
            // TODO: we probably should parse negTable, but it's not required until vocab changes are introduced. Since on the predefined vocab it will produce exact nTable, the same goes for expTable btw.
        }

        /*
                Since we're restoring vocab from previously serialized model, we can expect minWordFrequency appliance in its vocabulary, so it should NOT be truncated.
                That's why i'm setting minWordFrequency to configuration value, but applying SPECIAL to each word, to avoid truncation
         */
        VocabularyHolder holder = new VocabularyHolder.Builder()
                .minWordFrequency(configuration.getMinWordFrequency())
                .hugeModelExpected(configuration.isHugeModelExpected())
                .scavengerActivationThreshold(configuration.getScavengerActivationThreshold())
                .scavengerRetentionDelay(configuration.getScavengerRetentionDelay())
                .build();

        AtomicInteger counter = new AtomicInteger(0);
        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();
        while (iterator.hasNext()) {
            //    log.info("got line: " + iterator.nextSentence());
            String wordJson = iterator.nextSentence();
            VocabularyWord word = VocabularyWord.fromJson(wordJson);
            word.setSpecial(true);

            VocabWord vw = new VocabWord(word.getCount(), word.getWord());
            vw.setIndex(counter.getAndIncrement());

            vw.setIndex(word.getHuffmanNode().getIdx());
            vw.setCodeLength(word.getHuffmanNode().getLength());
            vw.setPoints(arrayToList(word.getHuffmanNode().getPoint(), word.getHuffmanNode().getLength()));
            vw.setCodes(arrayToList(word.getHuffmanNode().getCode(), word.getHuffmanNode().getLength()));

            vocabCache.addToken(vw);
            vocabCache.addWordToIndex(vw.getIndex(), vw.getLabel());
            vocabCache.putVocabWord(vw.getWord());
        }

        // at this moment vocab is restored, and it's time to rebuild Huffman tree
        // since word counters are equal, huffman tree will be equal too
        //holder.updateHuffmanCodes();

        // we definitely don't need UNK word in this scenarion


//        holder.transferBackToVocabCache(vocabCache, false);

        // now, it's time to transfer syn0/syn1/syn1 neg values
        InMemoryLookupTable lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
                .negative(configuration.getNegative())
                .useAdaGrad(configuration.isUseAdaGrad())
                .lr(configuration.getLearningRate())
                .cache(vocabCache)
                .vectorLength(configuration.getLayersSize())
                .build();

        // we create all arrays
        lookupTable.resetWeights(true);

        iterator.reset();

        // we should skip 3 lines from file
        iterator.nextSentence();
        iterator.nextSentence();
        iterator.nextSentence();

        // now, for each word from vocabHolder we'll just transfer actual values
        while (iterator.hasNext()) {
            String wordJson = iterator.nextSentence();
            VocabularyWord word = VocabularyWord.fromJson(wordJson);

            // syn0 transfer
            INDArray syn0 = lookupTable.getSyn0().getRow(vocabCache.indexOf(word.getWord()));
            syn0.assign(Nd4j.create(word.getSyn0()));

            // syn1 transfer
            // syn1 values are being accessed via tree points, but since our goal is just deserialization - we can just push it row by row
            INDArray syn1 =  lookupTable.getSyn1().getRow(vocabCache.indexOf(word.getWord()));
            syn1.assign(Nd4j.create(word.getSyn1()));

            // syn1Neg transfer
            if (configuration.getNegative() > 0) {
                INDArray syn1Neg =  lookupTable.getSyn1Neg().getRow(vocabCache.indexOf(word.getWord()));
                syn1Neg.assign(Nd4j.create(word.getSyn1Neg()));
            }
        }

        Word2Vec vec = new Word2Vec.Builder(configuration)
                .vocabCache(vocabCache)
                .lookupTable(lookupTable)
                .resetModel(false)
                .build();

        vec.setModelUtils(new BasicModelUtils());

        return vec;
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
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull String path)
            throws IOException {
        BufferedWriter write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(path), false), "UTF-8"));

        writeWordVectors(vec, write);

        write.flush();
        write.close();
    }

    /**
     * Writes the word vectors to the given path. Note that this assumes an in memory cache
     *
     * @param vec
     *            the word2vec to write
     * @param file
     *            the file to write
     * @throws IOException
     */
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull File file)
            throws IOException {
        BufferedWriter write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), "UTF-8"));

        writeWordVectors(vec, write);

        write.flush();
        write.close();
    }

    /**
     * Writes the word vectors to the given OutputStream. Note that this assumes an in memory cache.
     *
     * @param vec
     *            the word2vec to write
     * @param outputStream - OutputStream, where all data should be sent to
     *            the path to write
     * @throws IOException
     */
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull OutputStream outputStream) throws IOException {
        BufferedWriter writer =  new BufferedWriter(new OutputStreamWriter(outputStream, "UTF-8"));

        writeWordVectors(vec, writer);

        writer.flush();
        writer.close();
    }

    /**
     * Writes the word vectors to the given BufferedWriter. Note that this assumes an in memory cache.
     * BufferedWriter can be writer to local file, or hdfs file, or any compatible to java target.
     *
     * @param vec
     *            the word2vec to write
     * @param writer - BufferedWriter, where all data should be written to
     *            the path to write
     * @throws IOException
     */
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull BufferedWriter writer) throws IOException  {
        int words = 0;
        for (String word : vec.vocab().words()) {
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(word.replaceAll(" ", whitespaceReplacement));
            sb.append(" ");
            INDArray wordVector = vec.getWordVectorMatrix(word);
            for (int j = 0; j < wordVector.length(); j++) {
                sb.append(wordVector.getDouble(j));
                if (j < wordVector.length() - 1) {
                    sb.append(" ");
                }
            }
            sb.append("\n");
            writer.write(sb.toString());
            words++;
        }

        try {
            writer.flush();
        } catch (Exception e) {
            ;
        }
        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
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
        vectors.setModelUtils(new BasicModelUtils());
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
        vectors.setModelUtils(new BasicModelUtils());
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
            throws FileNotFoundException, UnsupportedEncodingException
    {
        Pair<InMemoryLookupTable, VocabCache> pair = loadTxt(vectorsFile);
        return fromPair(pair);
    }

    /**
     * Loads an in memory cache from the given path (sets syn0 and the vocab)
     *
     * @param vectorsFile the path of the file to load
     * @return a Pair holding the lookup table and the vocab cache.
     * @throws FileNotFoundException if the input file does not exist
     */
    public static Pair<InMemoryLookupTable, VocabCache> loadTxt(File vectorsFile)
            throws FileNotFoundException, UnsupportedEncodingException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(vectorsFile), "UTF-8"));
        VocabCache cache = new AbstractCache<>();

        LineIterator iter = IOUtils.lineIterator(reader);
        String line = null;
        boolean hasHeader = false;
        if (iter.hasNext()) {
            line = iter.nextLine();    // skip header line
            //look for spaces
            if(!line.contains(" ")) {
                log.info("Skipping first line");
                hasHeader = true;
            } else {
                // we should check for something that looks like proper word vectors here. i.e: 1 word at the 0 position, and bunch of floats further
                String[] split = line.split(" ");
                try {
                    for (int x = 1; x < split.length; x++) {
                        double val = Double.parseDouble(split[x]);
                    }
                    if (split.length < 4) hasHeader = true;
                } catch (Exception e) {
                    // if any conversion exception hits - that'll be considered header
                    hasHeader = true;

                    try {
                        reader.close();
                    } catch (Exception ex) {
                        ;
                    }
                }
            }

        }

        //reposition buffer to be one line ahead
        if(hasHeader) {
            line = "";
            iter.close();
            reader = new BufferedReader(new FileReader(vectorsFile));
            iter = IOUtils.lineIterator(reader);
            iter.nextLine();
        }

        List<INDArray> arrays = new ArrayList<>();
        while (iter.hasNext()) {
            if (line.isEmpty()) line = iter.nextLine();
            String[] split = line.split(" ");
            String word = split[0].replaceAll(whitespaceReplacement, " ");
            VocabWord word1 = new VocabWord(1.0, word);

            word1.setIndex(cache.numWords());

            cache.addToken(word1);

            cache.addWordToIndex(word1.getIndex(), word);

            cache.putVocabWord(word);
            INDArray row = Nd4j.create(Nd4j.createBuffer(split.length - 1));
            for (int i = 1; i < split.length; i++) {
                row.putScalar(i - 1, Float.parseFloat(split[i]));
            }
            arrays.add(row);

            // workaround for skipped first row
            line = "";
        }

        INDArray syn = Nd4j.create(new int[]{arrays.size(), arrays.get(0).columns()});
        for (int i = 0; i < syn.rows(); i++) {
            syn.putRow(i,arrays.get(i));
        }

        InMemoryLookupTable lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
                .vectorLength(arrays.get(0).columns())
                .useAdaGrad(false).cache(cache)
                .build();
        Nd4j.clearNans(syn);
        lookupTable.setSyn0(syn);

        iter.close();

        try {
            reader.close();
        } catch (Exception e) {
            ;
        }

        return new Pair<>(lookupTable, cache);
    }

    /**
     * This method can be used to load previously saved model from InputStream (like a HDFS-stream)
     *
     * @param stream InputStream that contains previously serialized model
     * @param skipFirstLine Set this TRUE if first line contains csv header, FALSE otherwise
     * @return
     * @throws IOException
     */
    public static WordVectors loadTxtVectors(@NonNull InputStream stream, boolean skipFirstLine) throws IOException {
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = "";
        List<INDArray> arrays = new ArrayList<>();

        if (skipFirstLine)
            reader.readLine();

        while((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            String word = split[0].replaceAll(whitespaceReplacement, " ");
            VocabWord word1 = new VocabWord(1.0, word);

            word1.setIndex(cache.numWords());

            cache.addToken(word1);

            cache.addWordToIndex(word1.getIndex(), word);

            cache.putVocabWord(word);
            INDArray row = Nd4j.create(Nd4j.createBuffer(split.length - 1));
            for (int i = 1; i < split.length; i++) {
                row.putScalar(i - 1, Float.parseFloat(split[i]));
            }
            arrays.add(row);
        }

        InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                .vectorLength(arrays.get(0).columns())
                .cache(cache)
                .build();

        INDArray syn = Nd4j.create(new int[]{arrays.size(), arrays.get(0).columns()});
        for (int i = 0; i < syn.rows(); i++) {
            syn.putRow(i,arrays.get(i));
        }

        Nd4j.clearNans(syn);
        lookupTable.setSyn0(syn);

        return fromPair(Pair.makePair((InMemoryLookupTable) lookupTable, (VocabCache) cache));
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
        BufferedWriter write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(csv), "UTF-8"));
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
            sb.append(word.replaceAll(" ", whitespaceReplacement));
            sb.append(" ");

            sb.append("\n");
            write.write(sb.toString());

        }

        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
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
        BufferedWriter write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(csv), "UTF-8"));
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
            sb.append(word.replaceAll(" ", whitespaceReplacement));
            sb.append(" ");

            sb.append("\n");
            write.write(sb.toString());

        }

        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
        write.flush();
        write.close();

    }


    /**
     *  This method is used only for VocabCache compatibility purposes
     * @param array
     * @param codeLen
     * @return
     */
    private static List<Integer> arrayToList(byte[] array, int codeLen) {
        List<Integer> result = new ArrayList<>();
        for (int x = 0; x < codeLen; x++) {
            result.add((int) array[x]);
        }
        return result;
    }

    private static byte[] listToArray(List<Integer> code) {
        byte[] array = new byte[40];
        for (int x = 0; x < code.size(); x++) {
            array[x]  = code.get(x).byteValue();
        }
        return array;
    }

    private static int[] listToArray(List<Integer> points, int codeLen) {
        int[] array = new int[points.size()];
        for (int x = 0; x < points.size(); x++) {
            array[x]  = points.get(x).intValue();
        }
        return array;
    }

    /**
     *  This method is used only for VocabCache compatibility purposes
     * @param array
     * @param codeLen
     * @return
     */
    private static List<Integer> arrayToList(int[] array, int codeLen) {
        List<Integer> result = new ArrayList<>();
        for (int x = 0; x < codeLen; x++) {
            result.add(array[x]);
        }
        return result;
    }

    /**
     * This method saves specified SequenceVectors model to target  file path
     *
     * @param vectors SequenceVectors model
     * @param factory SequenceElementFactory implementation for your objects
     * @param path Target output file path
     * @param <T>
     */
    public static <T extends  SequenceElement> void writeSequenceVectors(@NonNull SequenceVectors<T> vectors, @NonNull SequenceElementFactory<T> factory, @NonNull String path) throws IOException {
        writeSequenceVectors(vectors, factory, new FileOutputStream(path));
    }

    /**
     * This method saves specified SequenceVectors model to target  file
     *
     * @param vectors SequenceVectors model
     * @param factory SequenceElementFactory implementation for your objects
     * @param file Target output file
     * @param <T>
     */
    public static <T extends  SequenceElement> void writeSequenceVectors(@NonNull SequenceVectors<T> vectors, @NonNull SequenceElementFactory<T> factory,@NonNull  File file) throws IOException {
        writeSequenceVectors(vectors, factory, new FileOutputStream(file));
    }

    /**
     * This method saves specified SequenceVectors model to target  OutputStream
     *
     * @param vectors SequenceVectors model
     * @param factory SequenceElementFactory implementation for your objects
     * @param stream Target output stream
     * @param <T>
     */
    public static <T extends  SequenceElement> void writeSequenceVectors(@NonNull SequenceVectors<T> vectors, @NonNull SequenceElementFactory<T> factory, @NonNull OutputStream stream) throws IOException {
        WeightLookupTable<T> lookupTable = vectors.getLookupTable();
        VocabCache<T> vocabCache = vectors.getVocab();

        PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(stream, "UTF-8")));

        // at first line we save VectorsConfiguration
        writer.write(vectors.getConfiguration().toEncodedJson());

        // now we have elements one by one
        for (int x = 0; x < vocabCache.numWords();  x++) {
            T element = vocabCache.elementAtIndex(x);
            String json = factory.serialize(element);
            double[] vector = lookupTable.vector(element.getLabel()).data().asDouble();

            ElementPair pair = new ElementPair(json, vector);
            writer.println(pair.toEncodedJson());
        }
        writer.flush();
        writer.close();
    }

    /**
     * This method loads previously saved SequenceVectors model from File
     *
     * @param factory
     * @param file
     * @param <T>
     * @return
     */
    public static <T extends SequenceElement> SequenceVectors<T> readSequenceVectors(@NonNull SequenceElementFactory<T> factory, @NonNull File file) throws IOException {
        return readSequenceVectors(factory, new FileInputStream(file));
    }

    /**
     * This method loads previously saved SequenceVectors model from InputStream
     *
     * @param factory
     * @param stream
     * @param <T>
     * @return
     */
    public static <T extends SequenceElement> SequenceVectors<T> readSequenceVectors(@NonNull SequenceElementFactory<T> factory, @NonNull InputStream stream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream, "UTF-8"));

        // at first we load vectors configuration
        String line = reader.readLine();
        VectorsConfiguration configuration = VectorsConfiguration.fromJson(new String(Base64.decodeBase64(line),"UTF-8"));

        AbstractCache<T> vocabCache = new AbstractCache.Builder<T>().build();


        List<INDArray> rows = new ArrayList<>();
        while((line = reader.readLine()) != null) {
            ElementPair pair = ElementPair.fromEncodedJson(line);
            T element = factory.deserialize(pair.getObject());
            rows.add(Nd4j.create(pair.getVector()));

            vocabCache.addToken(element);
            vocabCache.addWordToIndex(element.getIndex(), element.getLabel());
        }

        reader.close();

        InMemoryLookupTable<T> lookupTable = (InMemoryLookupTable<T>)new InMemoryLookupTable.Builder<T>()
                .vectorLength(rows.get(0).columns())
                .build();

        INDArray syn0 = Nd4j.create(rows.size(), rows.get(0).columns());
        for (int x = 0; x < rows.size(); x++) {
            syn0.putRow(x, rows.get(x));
        }

        lookupTable.setSyn0(syn0);

        SequenceVectors<T> vectors = new SequenceVectors.Builder<T>(configuration)
                .vocabCache(vocabCache)
                .lookupTable(lookupTable)
                .resetModel(false)
                .build();

        return vectors;
    }

    public static void writeVocabCache(@NonNull VocabCache<VocabWord> vocabCache, @NonNull File file) throws FileNotFoundException, UnsupportedEncodingException {
        writeVocabCache(vocabCache, new FileOutputStream(file));
    }

    public static void writeVocabCache(@NonNull VocabCache<VocabWord> vocabCache, @NonNull OutputStream stream) throws UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(stream, "UTF-8")));

        for (int x = 0; x < vocabCache.numWords(); x++) {
            VocabWord word = vocabCache.elementAtIndex(x);
            writer.println(word.toJSON());
        }

        writer.flush();
        writer.close();
    }

    public static VocabCache<VocabWord> readVocabCache(@NonNull File file) throws IOException {
        return readVocabCache(new FileInputStream(file));
    }

    public static VocabCache<VocabWord> readVocabCache(@NonNull InputStream stream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream, "UTF-8"));
        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

        VocabWordFactory factory = new VocabWordFactory();

        String line = "";
        while((line = reader.readLine()) != null) {
            VocabWord word = factory.deserialize(line);

            vocabCache.addToken(word);
            vocabCache.addWordToIndex(word.getIndex(), word.getLabel());
        }

        return vocabCache;
    }

    /**
     * This is simple holder class
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    private static class ElementPair {
        private String object;
        private double[] vector;

        protected String toEncodedJson() {
            ObjectMapper mapper = SequenceElement.mapper();
            try {
                String json = mapper.writeValueAsString(this);
                String output = Base64.encodeBase64String(json.getBytes("UTF-8"));
                return output;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        protected static ElementPair fromEncodedJson(String encoded) {
            return null;
        }
    }
}
