/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.embeddings.loader;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.codec.binary.Base64;
import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.io.output.CloseShieldOutputStream;
import org.apache.commons.lang3.StringUtils;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.config.DL4JClassLoading;
import org.deeplearning4j.common.util.ND4JFileUtils;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.fasttext.FastText;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceElementFactory;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.serialization.VocabWordFactory;
import org.deeplearning4j.models.word2vec.StaticWord2Vec;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyHolder;
import org.deeplearning4j.models.word2vec.wordstore.VocabularyWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.OneTimeLogger;
import org.nd4j.compression.impl.NoOp;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.databind.DeserializationFeature;
import org.nd4j.shade.jackson.databind.MapperFeature;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.SerializationFeature;
import org.nd4j.storage.CompressedRamStorage;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

@Slf4j
public class WordVectorSerializer {
    private static final int MAX_SIZE = 50;
    private static final String WHITESPACE_REPLACEMENT = "_Az92_";

    private WordVectorSerializer() {
    }

    /**
     * Read a binary word2vec from input stream.
     *
     * @param inputStream input stream to read
     * @param linebreaks  if true, the reader expects each word/vector to be in a separate line, terminated
     *                    by a line break
     * @param normalize
     * @return a {@link Word2Vec model}
     * @throws NumberFormatException
     * @throws IOException
     * @throws FileNotFoundException
     */
    public static Word2Vec readBinaryModel(
            InputStream inputStream,
            boolean linebreaks,
            boolean normalize) throws NumberFormatException, IOException {
        InMemoryLookupTable<VocabWord> lookupTable;
        VocabCache<VocabWord> cache;
        INDArray syn0;
        int words, size;

        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();

        if (originalPeriodic)
            Nd4j.getMemoryManager().togglePeriodicGc(false);

        Nd4j.getMemoryManager().setOccasionalGcFrequency(50000);

        try (DataInputStream dis = new DataInputStream(inputStream)) {
            words = Integer.parseInt(ReadHelper.readString(dis));
            size = Integer.parseInt(ReadHelper.readString(dis));
            syn0 = Nd4j.create(words, size);
            cache = new AbstractCache<>();

            printOutProjectedMemoryUse(words, size, 1);

            lookupTable = new InMemoryLookupTable.Builder<VocabWord>()
                    .cache(cache)
                    .useHierarchicSoftmax(false)
                    .vectorLength(size)
                    .build();

            String word;
            float[] vector = new float[size];
            for (int i = 0; i < words; i++) {
                word = ReadHelper.readString(dis);
                log.trace("Loading {} with word {}", word, i);

                for (int j = 0; j < size; j++) {
                    vector[j] = ReadHelper.readFloat(dis);
                }

                if (cache.containsWord(word)) {
                    throw new ND4JIllegalStateException(
                            "Tried to add existing word. Probably time to switch linebreaks mode?");
                }

                syn0.putRow(i, normalize ? Transforms.unitVec(Nd4j.create(vector)) : Nd4j.create(vector));

                VocabWord vw = new VocabWord(1.0, word);
                vw.setIndex(cache.numWords());

                cache.addToken(vw);
                cache.addWordToIndex(vw.getIndex(), vw.getLabel());

                cache.putVocabWord(word);

                if (linebreaks) {
                    dis.readByte(); // line break
                }

                Nd4j.getMemoryManager().invokeGcOccasionally();
            }
        } finally {
            if (originalPeriodic) {
                Nd4j.getMemoryManager().togglePeriodicGc(true);
            }

            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);
        }

        lookupTable.setSyn0(syn0);

        Word2Vec ret = new Word2Vec
                .Builder()
                .useHierarchicSoftmax(false)
                .resetModel(false)
                .layerSize(syn0.columns())
                .allowParallelTokenization(true)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .learningRate(0.025)
                .windowSize(5)
                .workers(1)
                .vectorCalcThreads(1)
                .build();

        ret.setVocab(cache);
        ret.setLookupTable(lookupTable);

        return ret;
    }

    /**
     * This method writes word vectors to the given path.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param path
     * @param <T>
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable, String path)
            throws IOException {
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(path))) {
            writeWordVectors(lookupTable, bos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method writes word vectors to the given file.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param file
     * @param <T>
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable, File file)
            throws IOException {
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file))) {
            writeWordVectors(lookupTable, bos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method writes word vectors to the given OutputStream.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param stream
     * @param <T>
     * @throws IOException
     */
    public static <T extends SequenceElement> void writeWordVectors(WeightLookupTable<T> lookupTable,
                                                                    OutputStream stream) throws IOException {
        val vocabCache = lookupTable.getVocabCache();

        try (PrintWriter writer = new PrintWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8))) {
            // saving header as "NUM_WORDS VECTOR_SIZE NUM_DOCS"
            val str = vocabCache.numWords() + " " + lookupTable.layerSize() + " " + vocabCache.totalNumberOfDocs();
            log.debug("Saving header: {}", str);
            writer.println(str);

            // saving vocab content
            val num = vocabCache.numWords();
            for (int x = 0; x < num; x++) {
                T element = vocabCache.elementAtIndex(x);

                val builder = new StringBuilder();

                val l = element.getLabel();
                builder.append(ReadHelper.encodeB64(l)).append(" ");
                val vec = lookupTable.vector(element.getLabel());
                for (int i = 0; i < vec.length(); i++) {
                    builder.append(vec.getDouble(i));
                    if (i < vec.length() - 1)
                        builder.append(" ");
                }
                writer.println(builder);
            }
        }
    }


    /**
     * This method writes word vectors to the given OutputStream.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param stream
     * @param <T>
     * @throws IOException
     */
    public static <T extends SequenceElement> WeightLookupTable<T> readWordVectorsBinary(
            InputStream stream) throws IOException {
        InMemoryLookupTable.Builder<T> tableBuilder = new InMemoryLookupTable.Builder<>();
        InMemoryLookupTable<T> lookupTable;
        try (DataInputStream writer = new DataInputStream(stream)) {
            int numWords = writer.readInt();
            int layerSize = writer.readInt();
            long totalNumberOfDocs = writer.readLong();
            AbstractCache<T> vocabCache = new AbstractCache.Builder<T>().build();
            tableBuilder.vectorLength(layerSize)
                    .useAdaGrad(false);

            lookupTable = tableBuilder.build();
            // saving vocab content
            for (int x = 0; x < numWords; x++) {
                String label = writer.readUTF();
                boolean isLabel = writer.readBoolean();
                int codeLength = writer.readInt();
                List<Byte> codes = new ArrayList<>();
                for(int i = 0; i < codeLength; i++) {
                    codes.add((byte) writer.readInt());
                }

                int pointsLength = writer.readInt();
                int[] points = new int[pointsLength];
                for(int i = 0; i < pointsLength; i++) {
                    points[i] = writer.readInt();
                }


                VocabWord vocabWord = new VocabWord();
                vocabWord.setWord(label);
                vocabWord.setVocabId((long) x);
                vocabWord.setIndex(x);
                vocabWord.setPoints(points);
                vocabWord.setLabel(isLabel);
                vocabWord.setCodeLength((short) codeLength);
                vocabWord.setCodes(codes);
                long sequencesCount = writer.readLong();
                double elementFrequency = writer.readDouble();
                vocabWord.setSequencesCount(sequencesCount);
                vocabWord.setElementFrequency((long) elementFrequency);


                INDArray vec = Nd4j.createNpyFromInputStream(writer);

                vocabCache.addToken((T) vocabWord);

            }
        }

        return null;
    }



    /**
     * This method writes word vectors to the given OutputStream.
     * Please note: this method doesn't load whole vocab/lookupTable into memory, so it's able to process large vocabularies served over network.
     *
     * @param lookupTable
     * @param stream
     * @param <T>
     * @throws IOException
     */
    public static <T extends SequenceElement> void writeLookupTableBinary(InMemoryLookupTable<T> lookupTable,
                                                                          OutputStream stream) throws IOException {
        val vocabCache = lookupTable.getVocabCache();

        DataOutputStream writer = new DataOutputStream(stream);

        // saving header as "NUM_WORDS VECTOR_SIZE NUM_DOCS"
        val str = vocabCache.numWords() + " " + lookupTable.layerSize() + " " + vocabCache.totalNumberOfDocs();

        writer.writeInt(vocabCache.numWords());
        writer.writeInt(lookupTable.layerSize());
        writer.writeLong(vocabCache.totalNumberOfDocs());
        log.debug("Saving header: {}", str);


        // saving vocab content
        val num = vocabCache.numWords();
        for (int x = 0; x < num; x++) {
            T element = vocabCache.elementAtIndex(x);
            val l = element.getLabel();
            writer.writeUTF(l);
            writer.writeBoolean(element.isLabel());
            writer.writeInt(element.getCodeLength());
            for(int i = 0; i < element.getCodeLength(); i++) {
                writer.writeInt(element.getCodes().get(i));
            }

            writer.writeInt(element.getPoints().size());
            for(int i = 0; i < element.getPoints().size(); i++) {
                writer.writeInt(element.getPoints().get(i));
            }

            writer.writeLong(element.getSequencesCount());
            writer.writeDouble(element.getElementFrequency());

        }

        writer.writeBoolean(lookupTable.getSyn0() != null);

        if(lookupTable.getSyn0() != null) {
            //get length first so we can read later
            DataBuffer numpyPointer = Nd4j.convertToNumpy(lookupTable.getSyn0());
            writer.writeLong(numpyPointer.length());
            long written = Nd4j.writeAsNumpy(numpyPointer.pointer(),stream,false);
        }

        writer.writeBoolean(lookupTable.getSyn1() != null);
        if(lookupTable.getSyn1() != null) {
            DataBuffer numpyPointer = Nd4j.convertToNumpy(lookupTable.getSyn1());
            writer.writeLong(numpyPointer.capacity());
            long written = Nd4j.writeAsNumpy(numpyPointer.pointer(),stream,false);
        }

        writer.writeBoolean(lookupTable.getSyn1Neg() != null);
        if(lookupTable.getSyn1Neg() != null) {
            DataBuffer numpyPointer = Nd4j.convertToNumpy(lookupTable.getSyn1Neg());
            writer.writeLong(numpyPointer.capacity());
            long written = Nd4j.writeAsNumpy(numpyPointer.pointer(),stream,false);
        }

    }




    /**
     * This method saves paragraph vectors to the given file.
     *
     * @deprecated Use {@link #writeParagraphVectors(ParagraphVectors, File)}
     */
    @Deprecated
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
     * @deprecated Use {@link #writeParagraphVectors(ParagraphVectors, String)}
     */
    @Deprecated
    public static void writeWordVectors(@NonNull ParagraphVectors vectors, @NonNull String path) {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            writeWordVectors(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }



    /**
     * This method saves ParagraphVectors model into compressed zip file
     *
     * @param file
     */
    public static void writeParagraphVectors(ParagraphVectors vectors, File file) {
        try (FileOutputStream fos = new FileOutputStream(file);
             BufferedOutputStream stream = new BufferedOutputStream(fos)) {
            writeParagraphVectors(vectors, stream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves ParagraphVectors model into compressed zip file located at path
     *
     * @param path
     */
    public static void writeParagraphVectorsBinary(ParagraphVectors vectors, String path) {
        writeParagraphVectorsBinary(vectors, new File(path));
    }



    /**
     * This method restores Word2Vec model previously saved with writeWord2VecModel
     * <p>
     * PLEASE NOTE: This method loads FULL model, so don't use it if you're only going to use weights.
     *
     * @param file
     * @return
     */
    public static <T extends SequenceElement> SequenceVectors<T>  readSequenceVectorsBinary(File file) throws IOException {
        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();

        if (originalPeriodic)
            Nd4j.getMemoryManager().togglePeriodicGc(false);

        Nd4j.getMemoryManager().setOccasionalGcFrequency(50000);

        try {
            AbstractCache<VocabWord> vocabCache = new AbstractCache<>();

            try (DataInputStream reader = new DataInputStream(new FileInputStream(file))) {
                String json = reader.readUTF();
                VectorsConfiguration vectorsConfiguration = VectorsConfiguration.fromJson(json);

                // saving header as "NUM_WORDS VECTOR_SIZE NUM_DOCS"

                int numWords = reader.readInt();
                int layerSize = reader.readInt();
                long numDocs = reader.readLong();

                // saving vocab content
                for (int x = 0; x < numWords; x++) {
                    VocabWord vocabWord = new VocabWord();
                    String word = reader.readUTF();
                    boolean isLabel = reader.readBoolean();
                    int codeLength = reader.readInt();
                    List<Byte> codes = new ArrayList<>();
                    for(int i = 0; i < codeLength; i++) {
                        codes.add((byte) reader.readInt());
                    }

                    vocabWord.setCodes(codes);

                    int numPoints = reader.readInt();
                    List<Integer> points = new ArrayList<>();
                    for(int i = 0; i < numPoints; i++) {
                        points.add(reader.readInt());
                    }

                    vocabWord.setPoints(points);

                    long sequencesCount = reader.readLong();
                    double elementFrequency = reader.readDouble();
                    vocabWord.setSequencesCount(sequencesCount);
                    vocabWord.setElementFrequency((long) elementFrequency);
                    vocabWord.setWord(word);
                    vocabWord.setCodeLength((short) codes.size());
                    vocabCache.addToken(vocabWord);
                    vocabWord.setLabel(isLabel);
                    vocabWord.setIndex(x);
                    vocabCache.addWordToIndex(x,vocabWord.getWord());
                }

                vocabCache.setTotalDocCount(numDocs);
                boolean hasSyn0 = reader.readBoolean();
                INDArray syn0 = null;
                if(hasSyn0) {
                    long syn0Length = reader.readLong();
                    syn0 = Nd4j.createNpyFromInputStream(reader,syn0Length);
                }

                boolean hasSyn1 = reader.readBoolean();
                INDArray syn1 = null;
                if(hasSyn1) {
                    long syn1Length = reader.readLong();
                    syn1 = Nd4j.createNpyFromInputStream(reader,syn1Length);
                }

                boolean hasSyn1Neg = reader.readBoolean();
                INDArray syn1Neg = null;
                if(hasSyn1Neg) {
                    long syn1NegLong = reader.readLong();
                    syn1Neg = Nd4j.createNpyFromInputStream(reader,syn1NegLong);
                }

                vocabCache.setMinWordFrequency(vectorsConfiguration.getMinWordFrequency());
                vocabCache.setStopWords(new ArrayList<>(vectorsConfiguration.getStopList()));
                vocabCache.setScavengerThreshold(vectorsConfiguration.getScavengerActivationThreshold());
                vocabCache.setRetentionDelay(vectorsConfiguration.getScavengerRetentionDelay());

                InMemoryLookupTable<T> inMemoryLookupTable = new InMemoryLookupTable<>();
                inMemoryLookupTable.setSyn0(syn0);
                inMemoryLookupTable.setVectorLength(layerSize);
                inMemoryLookupTable.setSyn1(syn1);
                inMemoryLookupTable.setSyn1Neg(syn1Neg);
                inMemoryLookupTable.setVocab(vocabCache);
                SequenceVectors<T> sequenceVectors = new SequenceVectors.Builder<T>(vectorsConfiguration)
                        .lookupTable(inMemoryLookupTable)
                        .configuration(vectorsConfiguration)
                        .useHierarchicSoftmax(vectorsConfiguration.isUseHierarchicSoftmax())
                        .epochs(vectorsConfiguration.getEpochs())
                        .iterations(vectorsConfiguration.getIterations())
                        .layerSize(vectorsConfiguration.getLayersSize())
                        .batchSize(vectorsConfiguration.getBatchSize())
                        .learningRate(vectorsConfiguration.getLearningRate())
                        .windowSize(vectorsConfiguration.getWindow())
                        .minLearningRate(vectorsConfiguration.getMinLearningRate())
                        .minWordFrequency(vectorsConfiguration.getMinWordFrequency())
                        .stopWords(new ArrayList<>(vectorsConfiguration.getStopList()))
                        .negativeSample(vectorsConfiguration.getNegative())
                        .trainElementsRepresentation(vectorsConfiguration.isTrainElementsVectors())
                        .trainSequencesRepresentation(vectorsConfiguration.isTrainSequenceVectors())
                        .usePreciseMode(vectorsConfiguration.isPreciseMode())
                        .useUnknown(vectorsConfiguration.isUseUnknown())
                        .sequenceLearningAlgorithm(vectorsConfiguration.getSequenceLearningAlgorithm())
                        .elementsLearningAlgorithm(vectorsConfiguration.getElementsLearningAlgorithm())
                        .vocabCache((VocabCache<T>) vocabCache)
                        .batchSize(vectorsConfiguration.getBatchSize())
                        .windowSize(vectorsConfiguration.getWindow())
                        .build();
                return sequenceVectors;
            }


        } finally {
            if (originalPeriodic)
                Nd4j.getMemoryManager().togglePeriodicGc(true);

            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);

        }
    }

    /**
     * This method saves ParagraphVectors model into compressed zip file
     *
     * @param file
     */
    public static void writeParagraphVectorsBinary(ParagraphVectors vectors, File file) {
        try (FileOutputStream fos = new FileOutputStream(file)) {
            writeParagraphVectorsBinary(vectors, fos);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * This method restores ParagraphVectors model previously saved with writeParagraphVectors()
     *
     * @return
     */
    public static ParagraphVectors readParagraphVectorsBinary(File file) throws IOException {
        SequenceVectors<VocabWord> w2v = readSequenceVectorsBinary(file);

        // and "convert" it to ParaVec model + optionally trying to restore labels information
        ParagraphVectors vectors = new ParagraphVectors.Builder(w2v.getConfiguration())
                .vocabCache(w2v.getVocab())
                .lookupTable(w2v.lookupTable())
                .sequenceLearningAlgorithm(w2v.getConfiguration().getSequenceLearningAlgorithm())
                .elementsLearningAlgorithm(w2v.getConfiguration().getElementsLearningAlgorithm())
                .resetModel(false)
                .build();

        vectors.extractLabels();

        return vectors;
    }

    /**
     * This method saves ParagraphVectors model into compressed zip file and sends it to output stream
     */
    public static void writeSequenceVectorsBinary(SequenceVectors<VocabWord> vectors, OutputStream stream) throws IOException {

        try (DataOutputStream bufferedOutputStream = new DataOutputStream(stream)) {
            String encoded = vectors.getConfiguration().toJson();
            bufferedOutputStream.writeUTF(encoded);
            InMemoryLookupTable<VocabWord> weightLookupTable = (InMemoryLookupTable<VocabWord>) vectors.lookupTable();
            writeLookupTableBinary(weightLookupTable, bufferedOutputStream);
        } catch(Exception e) {
            e.printStackTrace();
        }

    }

    /**
     * This method saves ParagraphVectors model into compressed zip file and sends it to output stream
     */
    public static void writeParagraphVectorsBinary(ParagraphVectors vectors, OutputStream stream) throws IOException {
        writeSequenceVectorsBinary(vectors,stream);

    }


    /**
     * This method saves ParagraphVectors model into compressed zip file located at path
     *
     * @param path
     */
    public static void writeParagraphVectors(ParagraphVectors vectors, String path) {
        writeParagraphVectors(vectors, new File(path));
    }

    /**
     * This method saves Word2Vec model into compressed zip file
     * PLEASE NOTE: This method saves FULL model, including syn0 AND syn1
     */
    public static void writeWord2VecModel(Word2Vec vectors, File file) {
        try (FileOutputStream fos = new FileOutputStream(file);
             BufferedOutputStream stream = new BufferedOutputStream(fos)) {
            writeWord2VecModel(vectors, stream);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves Word2Vec model into compressed zip file
     * PLEASE NOTE: This method saves FULL model, including syn0 AND syn1
     */
    public static void writeWord2VecModel(Word2Vec vectors, String path) {
        writeWord2VecModel(vectors, new File(path));
    }

    /**
     * This method saves Word2Vec model into compressed zip file and sends it to output stream
     * PLEASE NOTE: This method saves FULL model, including syn0 AND syn1
     */
    public static void writeWord2VecModel(Word2Vec vectors, OutputStream stream) throws IOException {
        ZipOutputStream zipfile = new ZipOutputStream(new BufferedOutputStream(new CloseShieldOutputStream(stream)));

        ZipEntry syn0 = new ZipEntry("syn0.txt");
        zipfile.putNextEntry(syn0);

        // writing out syn0
        File tempFileSyn0 = ND4JFileUtils.createTempFile("word2vec", "0");
        File tempFileSyn1 = ND4JFileUtils.createTempFile("word2vec", "1");
        File tempFileSyn1Neg = ND4JFileUtils.createTempFile("word2vec", "n");
        File tempFileCodes = ND4JFileUtils.createTempFile("word2vec", "h");
        File tempFileHuffman = ND4JFileUtils.createTempFile("word2vec", "h");
        File tempFileFreqs = ND4JFileUtils.createTempFile("word2vec", "f");
        tempFileSyn0.deleteOnExit();
        tempFileSyn1.deleteOnExit();
        tempFileSyn1Neg.deleteOnExit();
        tempFileFreqs.deleteOnExit();
        tempFileCodes.deleteOnExit();
        tempFileHuffman.deleteOnExit();

        try {
            writeWordVectors(vectors.lookupTable(), tempFileSyn0);

            FileUtils.copyFile(tempFileSyn0, zipfile);

            // writing out syn1
            INDArray syn1 = ((InMemoryLookupTable<VocabWord>) vectors.getLookupTable()).getSyn1();

            if (syn1 != null)
                try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileSyn1))) {
                    for (int x = 0; x < syn1.rows(); x++) {
                        INDArray row = syn1.getRow(x);
                        StringBuilder builder = new StringBuilder();
                        for (int i = 0; i < row.length(); i++) {
                            builder.append(row.getDouble(i)).append(" ");
                        }
                        writer.println(builder.toString().trim());
                    }
                }

            ZipEntry zSyn1 = new ZipEntry("syn1.txt");
            zipfile.putNextEntry(zSyn1);

            FileUtils.copyFile(tempFileSyn1, zipfile);

            // writing out syn1
            INDArray syn1Neg = ((InMemoryLookupTable<VocabWord>) vectors.getLookupTable()).getSyn1Neg();

            if (syn1Neg != null)
                try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileSyn1Neg))) {
                    for (int x = 0; x < syn1Neg.rows(); x++) {
                        INDArray row = syn1Neg.getRow(x);
                        StringBuilder builder = new StringBuilder();
                        for (int i = 0; i < row.length(); i++) {
                            builder.append(row.getDouble(i)).append(" ");
                        }
                        writer.println(builder.toString().trim());
                    }
                }

            ZipEntry zSyn1Neg = new ZipEntry("syn1Neg.txt");
            zipfile.putNextEntry(zSyn1Neg);

            FileUtils.copyFile(tempFileSyn1Neg, zipfile);


            ZipEntry hC = new ZipEntry("codes.txt");
            zipfile.putNextEntry(hC);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileCodes))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(ReadHelper.encodeB64(word.getLabel())).append(" ");
                    for (int code : word.getCodes()) {
                        builder.append(code).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileCodes, zipfile);

            ZipEntry hP = new ZipEntry("huffman.txt");
            zipfile.putNextEntry(hP);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileHuffman))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(ReadHelper.encodeB64(word.getLabel())).append(" ");
                    for (int point : word.getPoints()) {
                        builder.append(point).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileHuffman, zipfile);

            ZipEntry hF = new ZipEntry("frequencies.txt");
            zipfile.putNextEntry(hF);

            // writing out word frequencies
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileFreqs))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(ReadHelper.encodeB64(word.getLabel())).append(" ")
                            .append(word.getElementFrequency()).append(" ")
                            .append(vectors.getVocab().docAppearedIn(word.getLabel()));

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileFreqs, zipfile);

            ZipEntry config = new ZipEntry("config.json");
            zipfile.putNextEntry(config);
            //log.info("Current config: {}", vectors.getConfiguration().toJson());
            try (ByteArrayInputStream bais = new ByteArrayInputStream(vectors.getConfiguration().toJson().getBytes(StandardCharsets.UTF_8))) {
                IOUtils.copy(bais, zipfile);
            }

            zipfile.flush();
            zipfile.close();
        } finally {
            for (File f : new File[]{tempFileSyn0, tempFileSyn1, tempFileSyn1Neg, tempFileCodes, tempFileHuffman, tempFileFreqs}) {
                try {
                    f.delete();
                } catch (Exception e) {
                    //Ignore, is temp file
                }
            }
        }
    }

    /**
     * This method saves ParagraphVectors model into compressed zip file and sends it to output stream
     */
    public static void writeParagraphVectors(ParagraphVectors vectors, OutputStream stream) throws IOException {
        ZipOutputStream zipfile = new ZipOutputStream(new BufferedOutputStream(new CloseShieldOutputStream(stream)));

        ZipEntry syn0 = new ZipEntry("syn0.txt");
        zipfile.putNextEntry(syn0);

        // writing out syn0
        File tempFileSyn0 = ND4JFileUtils.createTempFile("paravec", "0");
        File tempFileSyn1 = ND4JFileUtils.createTempFile("paravec", "1");
        File tempFileCodes = ND4JFileUtils.createTempFile("paravec", "h");
        File tempFileHuffman = ND4JFileUtils.createTempFile("paravec", "h");
        File tempFileFreqs = ND4JFileUtils.createTempFile("paravec", "h");
        tempFileSyn0.deleteOnExit();
        tempFileSyn1.deleteOnExit();
        tempFileCodes.deleteOnExit();
        tempFileHuffman.deleteOnExit();
        tempFileFreqs.deleteOnExit();

        try {

            writeWordVectors(vectors.lookupTable(), tempFileSyn0);

            FileUtils.copyFile(tempFileSyn0, zipfile);

            // writing out syn1
            INDArray syn1 = ((InMemoryLookupTable<VocabWord>) vectors.getLookupTable()).getSyn1();

            if (syn1 != null)
                try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileSyn1))) {
                    for (int x = 0; x < syn1.rows(); x++) {
                        INDArray row = syn1.getRow(x);
                        StringBuilder builder = new StringBuilder();
                        for (int i = 0; i < row.length(); i++) {
                            builder.append(row.getDouble(i)).append(" ");
                        }
                        writer.println(builder.toString().trim());
                    }
                }

            ZipEntry zSyn1 = new ZipEntry("syn1.txt");
            zipfile.putNextEntry(zSyn1);

            FileUtils.copyFile(tempFileSyn1, zipfile);

            ZipEntry hC = new ZipEntry("codes.txt");
            zipfile.putNextEntry(hC);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileCodes))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(ReadHelper.encodeB64(word.getLabel())).append(" ");
                    for (int code : word.getCodes()) {
                        builder.append(code).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileCodes, zipfile);

            ZipEntry hP = new ZipEntry("huffman.txt");
            zipfile.putNextEntry(hP);

            // writing out huffman tree
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileHuffman))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    StringBuilder builder = new StringBuilder(ReadHelper.encodeB64(word.getLabel())).append(" ");
                    for (int point : word.getPoints()) {
                        builder.append(point).append(" ");
                    }

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileHuffman, zipfile);

            ZipEntry config = new ZipEntry("config.json");
            zipfile.putNextEntry(config);
            IOUtils.write(vectors.getConfiguration().toJson(), zipfile, StandardCharsets.UTF_8);


            ZipEntry labels = new ZipEntry("labels.txt");
            zipfile.putNextEntry(labels);
            StringBuilder builder = new StringBuilder();
            for (VocabWord word : vectors.getVocab().tokens()) {
                if (word.isLabel())
                    builder.append(ReadHelper.encodeB64(word.getLabel())).append("\n");
            }
            IOUtils.write(builder.toString().trim(), zipfile, StandardCharsets.UTF_8);

            ZipEntry hF = new ZipEntry("frequencies.txt");
            zipfile.putNextEntry(hF);

            // writing out word frequencies
            try (PrintWriter writer = new PrintWriter(new FileWriter(tempFileFreqs))) {
                for (int i = 0; i < vectors.getVocab().numWords(); i++) {
                    VocabWord word = vectors.getVocab().elementAtIndex(i);
                    builder = new StringBuilder(ReadHelper.encodeB64(word.getLabel())).append(" ").append(word.getElementFrequency())
                            .append(" ").append(vectors.getVocab().docAppearedIn(word.getLabel()));

                    writer.println(builder.toString().trim());
                }
            }

            FileUtils.copyFile(tempFileFreqs, zipfile);

            zipfile.flush();
            zipfile.close();
        } finally {
            for (File f : new File[]{tempFileSyn0, tempFileSyn1, tempFileCodes, tempFileHuffman, tempFileFreqs}) {
                try {
                    f.delete();
                } catch (Exception e) {
                    //Ignore, is temp file
                }
            }
        }
    }


    /**
     * This method restores ParagraphVectors model previously saved with writeParagraphVectors()
     *
     * @return
     */
    public static ParagraphVectors readParagraphVectors(String path) throws IOException {
        return readParagraphVectors(new File(path));
    }


    /**
     * This method restores ParagraphVectors model previously saved with writeParagraphVectors()
     *
     * @return
     */
    public static  ParagraphVectors readParagraphVectors(File file) throws IOException {
      try(MemoryWorkspace workspace = Nd4j.getWorkspaceManager().createNewWorkspace()) {
          Word2Vec w2v = readWord2Vec(file);

          // and "convert" it to ParaVec model + optionally trying to restore labels information
          ParagraphVectors vectors = new ParagraphVectors.Builder(w2v.getConfiguration())
                  .vocabCache(w2v.getVocab())
                  .lookupTable(w2v.getLookupTable())
                  .resetModel(false)
                  .build();

          try (ZipFile zipFile = new ZipFile(file)) {
              // now we try to restore labels information
              ZipEntry labels = zipFile.getEntry("labels.txt");
              if (labels != null) {
                  InputStream stream = zipFile.getInputStream(labels);
                  try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
                      String line;
                      while ((line = reader.readLine()) != null) {
                          VocabWord word = vectors.getVocab().tokenFor(ReadHelper.decodeB64(line.trim()));
                          if (word != null) {
                              word.markAsLabel(true);
                          }
                      }
                  }
              }
          }

          vectors.extractLabels();

          return vectors;
      }

    }

    /**
     * This method restores Word2Vec model previously saved with writeWord2VecModel
     * <p>
     * PLEASE NOTE: This method loads FULL model, so don't use it if you're only going to use weights.
     *
     * @param file
     * @return
     * @deprecated Use {@link #readWord2Vec(File, boolean)}
     */
    @Deprecated
    public static  Word2Vec readWord2Vec(File file) throws IOException {
        File tmpFileSyn0 = ND4JFileUtils.createTempFile("word2vec", "0");
        File tmpFileSyn1 = ND4JFileUtils.createTempFile("word2vec", "1");
        File tmpFileC = ND4JFileUtils.createTempFile("word2vec", "c");
        File tmpFileH = ND4JFileUtils.createTempFile("word2vec", "h");
        File tmpFileF = ND4JFileUtils.createTempFile("word2vec", "f");

        tmpFileSyn0.deleteOnExit();
        tmpFileSyn1.deleteOnExit();
        tmpFileH.deleteOnExit();
        tmpFileC.deleteOnExit();
        tmpFileF.deleteOnExit();

        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();

        if (originalPeriodic)
            Nd4j.getMemoryManager().togglePeriodicGc(false);

        Nd4j.getMemoryManager().setOccasionalGcFrequency(50000);

        try {


            ZipFile zipFile = new ZipFile(file);
            ZipEntry syn0 = zipFile.getEntry("syn0.txt");
            InputStream stream = zipFile.getInputStream(syn0);

            FileUtils.copyInputStreamToFile(stream, tmpFileSyn0);

            ZipEntry syn1 = zipFile.getEntry("syn1.txt");
            stream = zipFile.getInputStream(syn1);

            FileUtils.copyInputStreamToFile(stream, tmpFileSyn1);

            ZipEntry codes = zipFile.getEntry("codes.txt");
            stream = zipFile.getInputStream(codes);

            FileUtils.copyInputStreamToFile(stream, tmpFileC);

            ZipEntry huffman = zipFile.getEntry("huffman.txt");
            stream = zipFile.getInputStream(huffman);

            FileUtils.copyInputStreamToFile(stream, tmpFileH);

            ZipEntry config = zipFile.getEntry("config.json");
            stream = zipFile.getInputStream(config);
            StringBuilder builder = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    builder.append(line);
                }
            }

            VectorsConfiguration configuration = VectorsConfiguration.fromJson(builder.toString().trim());

            // we read first 4 files as w2v model
            Word2Vec w2v = readWord2VecFromText(tmpFileSyn0, tmpFileSyn1, tmpFileC, tmpFileH, configuration);

            // we read frequencies from frequencies.txt, however it's possible that we might not have this file
            ZipEntry frequencies = zipFile.getEntry("frequencies.txt");
            if (frequencies != null) {
                stream = zipFile.getInputStream(frequencies);
                try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] split = line.split(" ");
                        VocabWord word = w2v.getVocab().tokenFor(ReadHelper.decodeB64(split[0]));
                        word.setElementFrequency((long) Double.parseDouble(split[1]));
                        word.setSequencesCount((long) Double.parseDouble(split[2]));
                    }
                }
            }


            ZipEntry zsyn1Neg = zipFile.getEntry("syn1Neg.txt");
            if (zsyn1Neg != null) {
                stream = zipFile.getInputStream(zsyn1Neg);

                try (InputStreamReader isr = new InputStreamReader(stream);
                     BufferedReader reader = new BufferedReader(isr)) {
                    String line = null;
                    List<INDArray> rows = new ArrayList<>();
                    while ((line = reader.readLine()) != null) {
                        String[] split = line.split(" ");
                        double array[] = new double[split.length];
                        for (int i = 0; i < split.length; i++) {
                            array[i] = Double.parseDouble(split[i]);
                        }

                        rows.add(Nd4j.create(array, new long[]{1,array.length},
                                ((InMemoryLookupTable) w2v.getLookupTable()).getSyn0().dataType()));
                    }

                    // it's possible to have full model without syn1Neg
                    if (!rows.isEmpty()) {
                        INDArray syn1Neg = Nd4j.vstack(rows);
                        ((InMemoryLookupTable) w2v.getLookupTable()).setSyn1Neg(syn1Neg);
                    }
                }
            }

            return w2v;
        } finally {
            if (originalPeriodic)
                Nd4j.getMemoryManager().togglePeriodicGc(true);

            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);

            for (File f : new File[]{tmpFileSyn0, tmpFileSyn1, tmpFileC, tmpFileH, tmpFileF}) {
                try {
                    f.delete();
                } catch (Exception e) {
                    //Ignore, is temp file
                }
            }
        }
    }

    /**
     * This method restores ParagraphVectors model previously saved with writeParagraphVectors()
     *
     * @return
     */
    public static ParagraphVectors readParagraphVectors(InputStream stream) throws IOException {
        File tmpFile = ND4JFileUtils.createTempFile("restore", "paravec");
        try {
            FileUtils.copyInputStreamToFile(stream, tmpFile);
            return readParagraphVectors(tmpFile);
        } finally {
            tmpFile.delete();
        }
    }

    /**
     * This method allows you to read ParagraphVectors from externally originated vectors and syn1.
     * So, technically this method is compatible with any other w2v implementation
     *
     * @param vectors  text file with words and their weights, aka Syn0
     * @param hs       text file HS layers, aka Syn1
     * @param h_codes  text file with Huffman tree codes
     * @param h_points text file with Huffman tree points
     * @return
     */
    public static Word2Vec readWord2VecFromText(@NonNull File vectors, @NonNull File hs, @NonNull File h_codes,
                                                @NonNull File h_points, @NonNull VectorsConfiguration configuration) throws IOException {
        // first we load syn0
        Pair<InMemoryLookupTable, VocabCache> pair = loadTxt(new FileInputStream(vectors));     //Note stream is closed in loadTxt
        InMemoryLookupTable lookupTable = pair.getFirst();
        lookupTable.setNegative(configuration.getNegative());
        if (configuration.getNegative() > 0)
            lookupTable.initNegative();
        VocabCache<VocabWord> vocab = (VocabCache<VocabWord>) pair.getSecond();

        // now we load syn1
        BufferedReader reader = new BufferedReader(new FileReader(hs));
        String line = null;
        List<INDArray> rows = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            double array[] = new double[split.length];
            for (int i = 0; i < split.length; i++) {
                array[i] = Double.parseDouble(split[i]);
            }

            DataBuffer typedBufferDetached = Nd4j.createTypedBufferDetached(array, lookupTable.getSyn0().dataType());

            INDArray arr = Nd4j.create(typedBufferDetached, new long[]{array.length}).detach();
            arr.setCloseable(false);
            rows.add(arr);
        }

        reader.close();

        // it's possible to have full model without syn1
        if (!rows.isEmpty()) {
            INDArray syn1 = Nd4j.vstack(rows);
            lookupTable.setSyn1(syn1);
        }

        // now we transform mappings into huffman tree points
        reader = new BufferedReader(new FileReader(h_points));
        while ((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            VocabWord word = vocab.wordFor(ReadHelper.decodeB64(split[0]));
            List<Integer> points = new ArrayList<>();
            for (int i = 1; i < split.length; i++) {
                points.add(Integer.parseInt(split[i]));
            }

            word.setPoints(points);
        }
        reader.close();


        // now we transform mappings into huffman tree codes
        reader = new BufferedReader(new FileReader(h_codes));
        while ((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            VocabWord word = vocab.wordFor(ReadHelper.decodeB64(split[0]));
            List<Byte> codes = new ArrayList<>();
            for (int i = 1; i < split.length; i++) {
                codes.add(Byte.parseByte(split[i]));
            }
            word.setCodes(codes);
            word.setCodeLength((short) codes.size());
        }
        reader.close();

        Word2Vec.Builder builder = new Word2Vec.Builder(configuration).vocabCache(vocab).lookupTable(lookupTable)
                .resetModel(false);

        TokenizerFactory factory = getTokenizerFactory(configuration);

        if (factory != null)
            builder.tokenizerFactory(factory);

        Word2Vec w2v = builder.build();

        return w2v;
    }


    /**
     * Restores previously serialized ParagraphVectors model
     * <p>
     * Deprecation note: Please, consider using readParagraphVectors() method instead
     *
     * @param path Path to file that contains previously serialized model
     * @return
     * @deprecated Use {@link #readParagraphVectors(String)}
     */
    @Deprecated
    public static ParagraphVectors readParagraphVectorsFromText(@NonNull String path) {
        return readParagraphVectorsFromText(new File(path));
    }

    /**
     * Restores previously serialized ParagraphVectors model
     * <p>
     * Deprecation note: Please, consider using readParagraphVectors() method instead
     *
     * @param file File that contains previously serialized model
     * @return
     * @deprecated Use {@link #readParagraphVectors(File)}
     */
    @Deprecated
    public static ParagraphVectors readParagraphVectorsFromText(@NonNull File file) {
        try (FileInputStream fis = new FileInputStream(file)) {
            return readParagraphVectorsFromText(fis);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * Restores previously serialized ParagraphVectors model
     * <p>
     * Deprecation note: Please, consider using readParagraphVectors() method instead
     *
     * @param stream InputStream that contains previously serialized model
     * @deprecated Use {@link #readParagraphVectors(InputStream)}
     */
    @Deprecated
    public static ParagraphVectors readParagraphVectorsFromText(@NonNull InputStream stream) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
            ArrayList<String> labels = new ArrayList<>();
            ArrayList<INDArray> arrays = new ArrayList<>();
            VocabCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();
            String line = "";
            while ((line = reader.readLine()) != null) {
                String[] split = line.split(" ");
                split[1] = split[1].replaceAll(WHITESPACE_REPLACEMENT, " ");
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
                } else
                    throw new IllegalStateException(
                            "Source stream doesn't looks like ParagraphVectors serialized model");

                // this particular line is just for backward compatibility with InMemoryLookupCache
                word.setIndex(vocabCache.numWords());

                vocabCache.addToken(word);
                vocabCache.addWordToIndex(word.getIndex(), word.getLabel());

                // backward compatibility code
                vocabCache.putVocabWord(word.getLabel());

                float[] vector = new float[split.length - 2];

                for (int i = 2; i < split.length; i++) {
                    vector[i - 2] = Float.parseFloat(split[i]);
                }

                INDArray row = Nd4j.create(vector);

                arrays.add(row);
            }

            // now we create syn0 matrix, using previously fetched rows
            /*INDArray syn = Nd4j.create(new int[]{arrays.size(), arrays.get(0).columns()});
            for (int i = 0; i < syn.rows(); i++) {
                syn.putRow(i, arrays.get(i));
            }*/
            INDArray syn = Nd4j.vstack(arrays);


            InMemoryLookupTable<VocabWord> lookupTable =
                    (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                            .vectorLength(arrays.get(0).columns()).useAdaGrad(false).cache(vocabCache)
                            .build();
            Nd4j.clearNans(syn);
            lookupTable.setSyn0(syn);

            LabelsSource source = new LabelsSource(labels);
            ParagraphVectors vectors = new ParagraphVectors.Builder().labelsSource(source).vocabCache(vocabCache)
                    .lookupTable(lookupTable).modelUtils(new BasicModelUtils<VocabWord>()).build();

            try {
                reader.close();
            } catch (Exception e) {
            }

            vectors.extractLabels();

            return vectors;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * This method saves paragraph vectors to the given output stream.
     *
     * @deprecated Use {@link #writeParagraphVectors(ParagraphVectors, OutputStream)}
     */
    @Deprecated
    public static void writeWordVectors(ParagraphVectors vectors, OutputStream stream) {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8))) {
            /*
            This method acts similarly to w2v csv serialization, except of additional tag for labels
             */

            VocabCache<VocabWord> vocabCache = vectors.getVocab();
            for (VocabWord word : vocabCache.vocabWords()) {
                StringBuilder builder = new StringBuilder();

                builder.append(word.isLabel() ? "L" : "E").append(" ");
                builder.append(word.getLabel().replaceAll(" ", WHITESPACE_REPLACEMENT)).append(" ");

                INDArray vector = vectors.getWordVectorMatrix(word.getLabel());
                for (int j = 0; j < vector.length(); j++) {
                    builder.append(vector.getDouble(j));
                    if (j < vector.length() - 1) {
                        builder.append(" ");
                    }
                }

                writer.write(builder.append("\n").toString());
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Writes the word vectors to the given path. Note that this assumes an in memory cache
     *
     * @param lookupTable
     * @param cache
     * @param path        the path to write
     * @throws IOException
     * @deprecated Use {@link #writeWord2VecModel(Word2Vec, File)} instead
     */
    @Deprecated
    public static void writeWordVectors(InMemoryLookupTable lookupTable, InMemoryLookupCache cache, String path)
            throws IOException {
        try (BufferedWriter write = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(path, false), StandardCharsets.UTF_8))) {
            for (int i = 0; i < lookupTable.getSyn0().rows(); i++) {
                String word = cache.wordAtIndex(i);
                if (word == null) {
                    continue;
                }
                StringBuilder sb = new StringBuilder();
                sb.append(word.replaceAll(" ", WHITESPACE_REPLACEMENT));
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
        }
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
     * <p>
     * Deprecation note: Please, consider using writeWord2VecModel() method instead
     *
     * @param vec  - The Word2Vec instance to be saved
     * @param path - the path for json to be saved
     * @deprecated Use writeWord2VecModel() method instead
     */
    @Deprecated
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


        if (!(lookupTable instanceof InMemoryLookupTable))
            throw new IllegalStateException("At this moment only InMemoryLookupTable is supported.");

        VectorsConfiguration conf = vec.getConfiguration();
        conf.setVocabSize(vocabCache.numWords());


        printWriter.println(conf.toJson());
        //log.info("Word2Vec conf. JSON: " + conf.toJson());
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
        } else
            printWriter.println("");


        List<VocabWord> words = new ArrayList<>(vocabCache.vocabWords());
        for (SequenceElement word : words) {
            VocabularyWord vw = new VocabularyWord(word.getLabel());
            vw.setCount(vocabCache.wordFrequency(word.getLabel()));

            vw.setHuffmanNode(VocabularyHolder.buildNode(word.getCodes(), word.getPoints(), word.getCodeLength(),
                    word.getIndex()));


            // writing down syn0
            INDArray syn0 = ((InMemoryLookupTable) lookupTable).getSyn0().getRow(vocabCache.indexOf(word.getLabel()));
            double[] dsyn0 = new double[syn0.columns()];
            for (int x = 0; x < conf.getLayersSize(); x++) {
                dsyn0[x] = syn0.getDouble(x);
            }
            vw.setSyn0(dsyn0);

            // writing down syn1
            INDArray syn1 = ((InMemoryLookupTable) lookupTable).getSyn1().getRow(vocabCache.indexOf(word.getLabel()));
            double[] dsyn1 = new double[syn1.columns()];
            for (int x = 0; x < syn1.columns(); x++) {
                dsyn1[x] = syn1.getDouble(x);
            }
            vw.setSyn1(dsyn1);

            // writing down syn1Neg, if negative sampling is used
            if (conf.getNegative() > 0 && ((InMemoryLookupTable) lookupTable).getSyn1Neg() != null) {
                INDArray syn1Neg = ((InMemoryLookupTable) lookupTable).getSyn1Neg()
                        .getRow(vocabCache.indexOf(word.getLabel()));
                double[] dsyn1Neg = new double[syn1Neg.columns()];
                for (int x = 0; x < syn1Neg.columns(); x++) {
                    dsyn1Neg[x] = syn1Neg.getDouble(x);
                }
                vw.setSyn1Neg(dsyn1Neg);
            }


            // in case of UseAdaGrad == true - we should save gradients for each word in vocab
            if (conf.isUseAdaGrad() && ((InMemoryLookupTable) lookupTable).isUseAdaGrad()) {
                INDArray gradient = word.getHistoricalGradient();
                if (gradient == null)
                    gradient = Nd4j.zeros(word.getCodes().size());
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
     * This method loads full w2v model, previously saved with writeFullMethod call
     * <p>
     * Deprecation note: Please, consider using readWord2VecModel() or loadStaticModel() method instead
     *
     * @param path - path to previously stored w2v json model
     * @return - Word2Vec instance
     * @deprecated Use readWord2VecModel() or loadStaticModel() method instead
     */
    @Deprecated
    public static Word2Vec loadFullModel(@NonNull String path) throws FileNotFoundException {
        /*
            // TODO: implementation is in process
            We need to restore:
                     1. WeightLookupTable, including syn0 and syn1 matrices
                     2. VocabCache + mark it as SPECIAL, to avoid accidental word removals
         */
        BasicLineIterator iterator = new BasicLineIterator(new File(path));

        // first 3 lines should be processed separately
        String confJson = iterator.nextSentence();
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
        VocabularyHolder holder = new VocabularyHolder.Builder().minWordFrequency(configuration.getMinWordFrequency())
                .hugeModelExpected(configuration.isHugeModelExpected())
                .scavengerActivationThreshold(configuration.getScavengerActivationThreshold())
                .scavengerRetentionDelay(configuration.getScavengerRetentionDelay()).build();

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
        InMemoryLookupTable lookupTable =
                (InMemoryLookupTable) new InMemoryLookupTable.Builder().negative(configuration.getNegative())
                        .useAdaGrad(configuration.isUseAdaGrad()).lr(configuration.getLearningRate())
                        .cache(vocabCache).vectorLength(configuration.getLayersSize()).build();

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
            INDArray syn1 = lookupTable.getSyn1().getRow(vocabCache.indexOf(word.getWord()));
            syn1.assign(Nd4j.create(word.getSyn1()));

            // syn1Neg transfer
            if (configuration.getNegative() > 0) {
                INDArray syn1Neg = lookupTable.getSyn1Neg().getRow(vocabCache.indexOf(word.getWord()));
                syn1Neg.assign(Nd4j.create(word.getSyn1Neg()));
            }
        }

        Word2Vec vec = new Word2Vec.Builder(configuration).vocabCache(vocabCache).lookupTable(lookupTable)
                .resetModel(false).build();

        vec.setModelUtils(new BasicModelUtils());

        return vec;
    }

    /**
     * Writes the word vectors to the given path. Note that this assumes an in memory cache
     *
     * @param vec  the word2vec to write
     * @param path the path to write
     * @deprecated Use {@link #writeWord2VecModel(Word2Vec, String)}
     */
    @Deprecated
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull String path) throws IOException {
        BufferedWriter write = new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(new File(path), false), "UTF-8"));

        writeWordVectors(vec, write);

        write.flush();
        write.close();
    }

    /**
     * Writes the word vectors to the given path. Note that this assumes an in memory cache
     *
     * @param vec  the word2vec to write
     * @param file the file to write
     * @deprecated Use {@link #writeWord2VecModel(Word2Vec, File)}
     */
    @Deprecated
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull File file) throws IOException {
        try (BufferedWriter write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8))) {
            writeWordVectors(vec, write);
        }
    }

    /**
     * Writes the word vectors to the given OutputStream. Note that this assumes an in memory cache.
     *
     * @param vec          the word2vec to write
     * @param outputStream - OutputStream, where all data should be sent to
     *                     the path to write
     * @deprecated Use {@link #writeWord2Vec(Word2Vec, OutputStream)}
     */
    @Deprecated
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull OutputStream outputStream) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(outputStream, StandardCharsets.UTF_8))) {
            writeWordVectors(vec, writer);
        }
    }

    /**
     * Writes the word vectors to the given BufferedWriter. Note that this assumes an in memory cache.
     * BufferedWriter can be writer to local file, or hdfs file, or any compatible to java target.
     *
     * @param vec    the word2vec to write
     * @param writer - BufferedWriter, where all data should be written to
     *               the path to write
     * @deprecated Use {@link #writeWord2Vec(Word2Vec, OutputStream)}
     */
    @Deprecated
    public static void writeWordVectors(@NonNull Word2Vec vec, @NonNull BufferedWriter writer) throws IOException {
        int words = 0;

        String str = vec.getVocab().numWords() + " " + vec.getLayerSize() + " " + vec.getVocab().totalNumberOfDocs();
        log.debug("Saving header: {}", str);
        writer.write(str + "\n");

        for (String word : vec.vocab().words()) {
            if (word == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            sb.append(word.replaceAll(" ", WHITESPACE_REPLACEMENT));
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
        }
        log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
    }

    /**
     * Load word vectors for the given vocab and table
     *
     * @param table the weights to use
     * @param vocab the vocab to use
     * @return wordvectors based on the given parameters
     */
    public static WordVectors fromTableAndVocab(WeightLookupTable table, VocabCache vocab) {
        WordVectorsImpl vectors = new WordVectorsImpl();
        vectors.setLookupTable(table);
        vectors.setVocab(vocab);
        vectors.setModelUtils(new BasicModelUtils());
        return vectors;
    }

    /**
     * Load word vectors from the given pair
     *
     * @param pair the given pair
     * @return a read only word vectors impl based on the given lookup table and vocab
     */
    public static Word2Vec fromPair(Pair<InMemoryLookupTable, VocabCache> pair) {
        Word2Vec vectors = new Word2Vec();
        vectors.setLookupTable(pair.getFirst());
        vectors.setVocab(pair.getSecond());
        vectors.setModelUtils(new BasicModelUtils());
        return vectors;
    }

    /**
     * Loads an in memory cache from the given path (sets syn0 and the vocab)
     * <p>
     * Deprecation note: Please, consider using readWord2VecModel() or loadStaticModel() method instead
     *
     * @param vectorsFile the path of the file to load\
     * @return
     * @throws FileNotFoundException if the file does not exist
     * @deprecated Use {@link #loadTxt(InputStream)}
     */
    @Deprecated
    public static WordVectors loadTxtVectors(File vectorsFile) throws IOException {
        FileInputStream fileInputStream = new FileInputStream(vectorsFile);         //Note stream is closed in loadTxt
        Pair<InMemoryLookupTable, VocabCache> pair = loadTxt(fileInputStream);
        return fromPair(pair);
    }

    static InputStream fileStream(@NonNull File file) throws IOException {
        boolean isZip = file.getName().endsWith(".zip");
        boolean isGzip = GzipUtils.isCompressedFilename(file.getName());

        InputStream inputStream;

        if (isZip) {
            inputStream = decompressZip(file);
        } else if (isGzip) {
            FileInputStream fis = new FileInputStream(file);
            inputStream = new GZIPInputStream(fis);
        } else {
            inputStream = new FileInputStream(file);
        }

        return new BufferedInputStream(inputStream);
    }

    private static InputStream decompressZip(File modelFile) throws IOException {
        ZipFile zipFile = new ZipFile(modelFile);
        InputStream inputStream = null;

        try (FileInputStream fis = new FileInputStream(modelFile);
             BufferedInputStream bis = new BufferedInputStream(fis);
             ZipInputStream zipStream = new ZipInputStream(bis)) {
            ZipEntry entry;
            if ((entry = zipStream.getNextEntry()) != null) {
                inputStream = zipFile.getInputStream(entry);
            }

            if (zipStream.getNextEntry() != null) {
                throw new RuntimeException("Zip archive " + modelFile + " contains more than 1 file");
            }
        }

        return inputStream;
    }

    public static Pair<InMemoryLookupTable, VocabCache> loadTxt(@NonNull File file) {
        try (InputStream inputStream = fileStream(file)) {
            return loadTxt(inputStream);
        } catch (IOException readTestException) {
            throw new RuntimeException(readTestException);
        }
    }

    /**
     * Loads an in memory cache from the given input stream (sets syn0 and the vocab).
     *
     * @param inputStream  input stream
     * @return a {@link Pair} holding the lookup table and the vocab cache.
     */
    @SneakyThrows
    public static Pair<InMemoryLookupTable, VocabCache> loadTxt(@NonNull InputStream inputStream) {
        AbstractCache<VocabWord> cache = new AbstractCache<>();
        LineIterator lines = null;

        try (InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
             BufferedReader reader = new BufferedReader(inputStreamReader)) {
            lines = IOUtils.lineIterator(reader);

            String line = null;
            boolean hasHeader = false;

            /* Check if first line is a header */
            if (lines.hasNext()) {
                line = lines.nextLine();
                hasHeader = isHeader(line, cache);
            }

            if (hasHeader) {
                log.debug("First line is a header");
                line = lines.nextLine();
            }

            List<INDArray> arrays = new ArrayList<>();
            long[] vShape = new long[]{ 1, -1 };

            do {
                String[] tokens = line.split(" ");
                String word = ReadHelper.decodeB64(tokens[0]);
                VocabWord vocabWord = new VocabWord(1.0, word);
                vocabWord.setIndex(cache.numWords());

                cache.addToken(vocabWord);
                cache.addWordToIndex(vocabWord.getIndex(), word);
                cache.putVocabWord(word);

                float[] vector = new float[tokens.length - 1];
                for (int i = 1; i < tokens.length; i++) {
                    vector[i - 1] = Float.parseFloat(tokens[i]);
                }

                vShape[1] = vector.length;
                INDArray row = Nd4j.create(vector, vShape);

                arrays.add(row);

                line = lines.hasNext() ? lines.next() : null;
            } while (line != null);

            INDArray syn = Nd4j.vstack(arrays);

            InMemoryLookupTable<VocabWord> lookupTable = new InMemoryLookupTable
                    .Builder<VocabWord>()
                    .vectorLength(arrays.get(0).columns())
                    .useAdaGrad(false)
                    .cache(cache)
                    .useHierarchicSoftmax(false)
                    .build();

            lookupTable.setSyn0(syn);

            return new Pair<>((InMemoryLookupTable) lookupTable, (VocabCache) cache);
        } catch (IOException readeTextStreamException) {
            throw new RuntimeException(readeTextStreamException);
        } finally {
            if (lines != null) {
                lines.close();
            }
        }
    }

    static boolean isHeader(String line, AbstractCache cache) {
        if (!line.contains(" ")) {
            return true;
        } else {

            /* We should check for something that looks like proper word vectors here. i.e: 1 word at the 0
             * position, and bunch of floats further */
            String[] headers = line.split(" ");

            try {
                long[] header = new long[headers.length];
                for (int x = 0; x < headers.length; x++) {
                    header[x] = Long.parseLong(headers[x]);
                }

                /* Now we know, if that's all ints - it's just a header
                 * [0] - number of words
                 * [1] - vectorLength
                 * [2] - number of documents <-- DL4j-only value
                 */
                if (headers.length == 3) {
                    long numberOfDocuments = header[2];
                    cache.incrementTotalDocCount(numberOfDocuments);
                }

                long numWords = header[0];
                int vectorLength = (int) header[1];
                printOutProjectedMemoryUse(numWords, vectorLength, 1);

                return true;
            } catch (Exception notHeaderException) {
                // if any conversion exception hits - that'll be considered header
                return false;
            }
        }
    }

    /**
     * This method can be used to load previously saved model from InputStream (like a HDFS-stream)
     * <p>
     * Deprecation note: Please, consider using readWord2VecModel() or loadStaticModel() method instead
     *
     * @param stream        InputStream that contains previously serialized model
     * @param skipFirstLine Set this TRUE if first line contains csv header, FALSE otherwise
     * @return
     * @throws IOException
     * @deprecated Use readWord2VecModel() or loadStaticModel() method instead
     */
    @Deprecated
    public static WordVectors loadTxtVectors(@NonNull InputStream stream, boolean skipFirstLine) throws IOException {
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String line = "";
        List<INDArray> arrays = new ArrayList<>();

        if (skipFirstLine)
            reader.readLine();

        while ((line = reader.readLine()) != null) {
            String[] split = line.split(" ");
            String word = split[0].replaceAll(WHITESPACE_REPLACEMENT, " ");
            VocabWord word1 = new VocabWord(1.0, word);

            word1.setIndex(cache.numWords());

            cache.addToken(word1);

            cache.addWordToIndex(word1.getIndex(), word);

            cache.putVocabWord(word);

            float[] vector = new float[split.length - 1];

            for (int i = 1; i < split.length; i++) {
                vector[i - 1] = Float.parseFloat(split[i]);
            }

            INDArray row = Nd4j.create(vector);

            arrays.add(row);
        }

        InMemoryLookupTable<VocabWord> lookupTable =
                (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                        .vectorLength(arrays.get(0).columns()).cache(cache).build();

        INDArray syn = Nd4j.vstack(arrays);

        Nd4j.clearNans(syn);
        lookupTable.setSyn0(syn);

        return fromPair(Pair.makePair((InMemoryLookupTable) lookupTable, (VocabCache) cache));
    }

    /**
     * Write the tsne format
     *
     * @param vec  the word vectors to use for labeling
     * @param tsne the tsne array to write
     * @param csv  the file to use
     * @throws Exception
     */
    public static void writeTsneFormat(Word2Vec vec, INDArray tsne, File csv) throws Exception {
        try (BufferedWriter write = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(csv), StandardCharsets.UTF_8))) {
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
                sb.append(word.replaceAll(" ", WHITESPACE_REPLACEMENT));
                sb.append(" ");

                sb.append("\n");
                write.write(sb.toString());

            }

            log.info("Wrote " + words + " with size of " + vec.lookupTable().layerSize());
        }
    }


    /**
     * This method is used only for VocabCache compatibility purposes
     *
     * @param array
     * @param codeLen
     * @return
     */
    private static List<Byte> arrayToList(byte[] array, int codeLen) {
        List<Byte> result = new ArrayList<>();
        for (int x = 0; x < codeLen; x++) {
            result.add(array[x]);
        }
        return result;
    }

    private static byte[] listToArray(List<Byte> code) {
        byte[] array = new byte[40];
        for (int x = 0; x < code.size(); x++) {
            array[x] = code.get(x).byteValue();
        }
        return array;
    }

    private static int[] listToArray(List<Integer> points, int codeLen) {
        int[] array = new int[points.size()];
        for (int x = 0; x < points.size(); x++) {
            array[x] = points.get(x).intValue();
        }
        return array;
    }

    /**
     * This method is used only for VocabCache compatibility purposes
     *
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
     * @param path    Target output file path
     * @param <T>
     */
    public static <T extends SequenceElement> void writeSequenceVectors(@NonNull SequenceVectors<T> vectors,
                                                                        @NonNull SequenceElementFactory<T> factory, @NonNull String path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            writeSequenceVectors(vectors, factory, fos);
        }
    }

    /**
     * This method saves specified SequenceVectors model to target  file
     *
     * @param vectors SequenceVectors model
     * @param factory SequenceElementFactory implementation for your objects
     * @param file    Target output file
     * @param <T>
     */
    public static <T extends SequenceElement> void writeSequenceVectors(@NonNull SequenceVectors<T> vectors,
                                                                        @NonNull SequenceElementFactory<T> factory, @NonNull File file) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(file)) {
            writeSequenceVectors(vectors, factory, fos);
        }
    }

    /**
     * This method saves specified SequenceVectors model to target  OutputStream
     *
     * @param vectors SequenceVectors model
     * @param factory SequenceElementFactory implementation for your objects
     * @param stream  Target output stream
     * @param <T>
     */
    public static <T extends SequenceElement> void writeSequenceVectors(@NonNull SequenceVectors<T> vectors,
                                                                        @NonNull SequenceElementFactory<T> factory, @NonNull OutputStream stream) throws IOException {
        WeightLookupTable<T> lookupTable = vectors.getLookupTable();
        VocabCache<T> vocabCache = vectors.getVocab();

        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8)))) {

            // at first line we save VectorsConfiguration
            writer.write(vectors.getConfiguration().toEncodedJson());

            // now we have elements one by one
            for (int x = 0; x < vocabCache.numWords(); x++) {
                T element = vocabCache.elementAtIndex(x);
                String json = factory.serialize(element);
                INDArray d = Nd4j.create(1);
                double[] vector = lookupTable.vector(element.getLabel()).dup().data().asDouble();
                ElementPair pair = new ElementPair(json, vector);
                writer.println(pair.toEncodedJson());
                writer.flush();
            }
        }
    }

    private static final String CONFIG_ENTRY = "config.json";
    private static final String VOCAB_ENTRY = "vocabulary.json";
    private static final String SYN0_ENTRY = "syn0.bin";
    private static final String SYN1_ENTRY = "syn1.bin";
    private static final String SYN1_NEG_ENTRY = "syn1neg.bin";

    /**
     * This method saves specified SequenceVectors model to target  OutputStream
     *
     * @param vectors SequenceVectors model
     * @param stream  Target output stream
     * @param <T>
     */
    public static <T extends SequenceElement> void writeSequenceVectors(@NonNull SequenceVectors<T> vectors,
                                                                        @NonNull OutputStream stream)
            throws IOException {

        InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>) vectors.getLookupTable();
        AbstractCache<T> vocabCache = (AbstractCache<T>) vectors.getVocab();

        try (ZipOutputStream zipfile = new ZipOutputStream(new BufferedOutputStream(new CloseShieldOutputStream(stream)));
             DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(zipfile))) {

            ZipEntry config = new ZipEntry(CONFIG_ENTRY);
            zipfile.putNextEntry(config);
            VectorsConfiguration configuration = vectors.getConfiguration();

            String json = configuration.toJson().trim();
            zipfile.write(json.getBytes("UTF-8"));

            ZipEntry vocab = new ZipEntry(VOCAB_ENTRY);
            zipfile.putNextEntry(vocab);
            zipfile.write(vocabCache.toJson().getBytes("UTF-8"));

            INDArray syn0Data = lookupTable.getSyn0();
            ZipEntry syn0 = new ZipEntry(SYN0_ENTRY);
            zipfile.putNextEntry(syn0);
            Nd4j.write(syn0Data, dos);
            dos.flush();

            INDArray syn1Data = lookupTable.getSyn1();
            if (syn1Data != null) {
                ZipEntry syn1 = new ZipEntry(SYN1_ENTRY);
                zipfile.putNextEntry(syn1);
                Nd4j.write(syn1Data, dos);
                dos.flush();
            }

            INDArray syn1NegData = lookupTable.getSyn1Neg();
            if (syn1NegData != null) {
                ZipEntry syn1neg = new ZipEntry(SYN1_NEG_ENTRY);
                zipfile.putNextEntry(syn1neg);
                Nd4j.write(syn1NegData, dos);
                dos.flush();
            }
        }
    }

    /**
     * This method loads SequenceVectors from specified file path
     *
     * @param path String
     * @param readExtendedTables boolean
     * @param <T>
     */
    public static <T extends SequenceElement> SequenceVectors<T> readSequenceVectors(@NonNull String path,
                                                                                     boolean readExtendedTables)
            throws IOException {

        File file = new File(path);
        SequenceVectors<T> vectors = readSequenceVectors(file, readExtendedTables);
        return vectors;
    }

    /**
     * This method loads SequenceVectors from specified file path
     *
     * @param file File
     * @param readExtendedTables boolean
     * @param <T>
     */

    public static <T extends SequenceElement> SequenceVectors<T> readSequenceVectors(@NonNull File file,
                                                                                     boolean readExtendedTables)
            throws IOException {

        SequenceVectors<T> vectors = readSequenceVectors(new FileInputStream(file), readExtendedTables);
        return vectors;
    }

    /**
     * This method loads SequenceVectors from specified input stream
     *
     * @param stream InputStream
     * @param readExtendedTables boolean
     * @param <T>
     */
    public static <T extends SequenceElement> SequenceVectors<T> readSequenceVectors(@NonNull InputStream stream,
                                                                                     boolean readExtendedTables)
            throws IOException {

        SequenceVectors<T> vectors = null;
        AbstractCache<T> vocabCache = null;
        VectorsConfiguration configuration = null;

        INDArray syn0 = null, syn1 = null, syn1neg = null;

        try (ZipInputStream zipfile = new ZipInputStream(new BufferedInputStream(stream))) {

            ZipEntry entry = null;
            while ((entry = zipfile.getNextEntry()) != null) {

                String name = entry.getName();
                byte[] bytes = IOUtils.toByteArray(zipfile);

                if (name.equals(CONFIG_ENTRY)) {
                    String content = new String(bytes, "UTF-8");
                    configuration = VectorsConfiguration.fromJson(content);
                    continue;
                } else if (name.equals(VOCAB_ENTRY)) {
                    String content = new String(bytes, "UTF-8");
                    vocabCache = AbstractCache.fromJson(content);
                    continue;
                }
                if (readExtendedTables) {
                    if (name.equals(SYN0_ENTRY)) {
                        syn0 = Nd4j.read(new ByteArrayInputStream(bytes));

                    } else if (name.equals(SYN1_ENTRY)) {
                        syn1 = Nd4j.read(new ByteArrayInputStream(bytes));
                    } else if (name.equals(SYN1_NEG_ENTRY)) {
                        syn1neg = Nd4j.read(new ByteArrayInputStream(bytes));
                    }
                }
            }

        }
        InMemoryLookupTable<T> lookupTable = new InMemoryLookupTable<>();
        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);
        lookupTable.setSyn1Neg(syn1neg);
        vectors = new SequenceVectors.Builder<T>(configuration).
                lookupTable(lookupTable).
                vocabCache(vocabCache).
                build();
        return vectors;
    }

    /**
     * This method loads previously saved SequenceVectors model from File
     *
     * @param factory
     * @param file
     * @param <T>
     * @return
     */
    public static <T extends SequenceElement> SequenceVectors<T> readSequenceVectors(
            @NonNull SequenceElementFactory<T> factory, @NonNull File file) throws IOException {
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
    public static <T extends SequenceElement> SequenceVectors<T> readSequenceVectors(
            @NonNull SequenceElementFactory<T> factory, @NonNull InputStream stream) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream, "UTF-8"));

        // at first we load vectors configuration
        String line = reader.readLine();
        VectorsConfiguration configuration =
                VectorsConfiguration.fromJson(new String(Base64.decodeBase64(line), "UTF-8"));

        AbstractCache<T> vocabCache = new AbstractCache.Builder<T>().build();


        List<INDArray> rows = new ArrayList<>();

        while ((line = reader.readLine()) != null) {
            if (line.isEmpty()) // skip empty line
                continue;
            ElementPair pair = ElementPair.fromEncodedJson(line);
            T element = factory.deserialize(pair.getObject());
            rows.add(Nd4j.create(pair.getVector()));
            vocabCache.addToken(element);
            vocabCache.addWordToIndex(element.getIndex(), element.getLabel());
        }

        reader.close();

        InMemoryLookupTable<T> lookupTable = (InMemoryLookupTable<T>) new InMemoryLookupTable.Builder<T>()
                .vectorLength(rows.get(0).columns()).cache(vocabCache).build(); // fix: add vocab cache

        /*
         * INDArray syn0 = Nd4j.create(rows.size(), rows.get(0).columns()); for (int x = 0; x < rows.size(); x++) {
         * syn0.putRow(x, rows.get(x)); }
         */
        INDArray syn0 = Nd4j.vstack(rows);

        lookupTable.setSyn0(syn0);

        SequenceVectors<T> vectors = new SequenceVectors.Builder<T>(configuration).vocabCache(vocabCache)
                .lookupTable(lookupTable).resetModel(false).build();

        return vectors;
    }

    /**
     * This method saves vocab cache to provided File.
     * Please note: it saves only vocab content, so it's suitable mostly for BagOfWords/TF-IDF vectorizers
     *
     * @param vocabCache
     * @param file
     * @throws UnsupportedEncodingException
     */
    public static void writeVocabCache(@NonNull VocabCache<VocabWord> vocabCache, @NonNull File file)
            throws IOException {
        try (FileOutputStream fos = new FileOutputStream(file)) {
            writeVocabCache(vocabCache, fos);
        }
    }

    /**
     * This method saves vocab cache to provided OutputStream.
     * Please note: it saves only vocab content, so it's suitable mostly for BagOfWords/TF-IDF vectorizers
     *
     * @param vocabCache
     * @param stream
     * @throws UnsupportedEncodingException
     */
    public static void writeVocabCache(@NonNull VocabCache<VocabWord> vocabCache, @NonNull OutputStream stream)
            throws IOException {
        try (PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(stream, StandardCharsets.UTF_8)))) {
            // saving general vocab information
            writer.println("" + vocabCache.numWords() + " " + vocabCache.totalNumberOfDocs() + " " + vocabCache.totalWordOccurrences());

            for (int x = 0; x < vocabCache.numWords(); x++) {
                VocabWord word = vocabCache.elementAtIndex(x);
                writer.println(word.toJSON());
            }
        }
    }

    /**
     * This method reads vocab cache from provided file.
     * Please note: it reads only vocab content, so it's suitable mostly for BagOfWords/TF-IDF vectorizers
     *
     * @param file
     * @return
     * @throws IOException
     */
    public static VocabCache<VocabWord> readVocabCache(@NonNull File file) throws IOException {
        try (FileInputStream fis = new FileInputStream(file)) {
            return readVocabCache(fis);
        }
    }

    /**
     * This method reads vocab cache from provided InputStream.
     * Please note: it reads only vocab content, so it's suitable mostly for BagOfWords/TF-IDF vectorizers
     *
     * @param stream
     * @return
     * @throws IOException
     */
    public static VocabCache<VocabWord> readVocabCache(@NonNull InputStream stream) throws IOException {
        val vocabCache = new AbstractCache.Builder<VocabWord>().build();
        val factory = new VocabWordFactory();
        boolean firstLine = true;
        long totalWordOcc = -1L;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // try to treat first line as header with 3 digits
                if (firstLine) {
                    firstLine = false;
                    val split = line.split("\\ ");

                    if (split.length != 3)
                        continue;

                    try {
                        vocabCache.setTotalDocCount(Long.valueOf(split[1]));
                        totalWordOcc = Long.valueOf(split[2]);
                        continue;
                    } catch (NumberFormatException e) {
                        // no-op
                    }
                }

                val word = factory.deserialize(line);

                vocabCache.addToken(word);
                vocabCache.addWordToIndex(word.getIndex(), word.getLabel());
            }
        }

        if (totalWordOcc >= 0)
            vocabCache.setTotalWordOccurences(totalWordOcc);

        return vocabCache;
    }

    /**
     * This is utility holder class
     */
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    private static class ElementPair {
        private String object;
        private double[] vector;

        /**
         * This utility method serializes ElementPair into JSON + packs it into Base64-encoded string
         *
         * @return
         */
        protected String toEncodedJson() {
            ObjectMapper mapper = SequenceElement.mapper();
            Base64 base64 = new Base64(Integer.MAX_VALUE);
            try {
                String json = mapper.writeValueAsString(this);
                String output = base64.encodeAsString(json.getBytes("UTF-8"));
                return output;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        /**
         * This utility method returns ElementPair from Base64-encoded string
         *
         * @param encoded
         * @return
         */
        protected static ElementPair fromEncodedJson(String encoded) {
            ObjectMapper mapper = SequenceElement.mapper();
            try {
                String decoded = new String(Base64.decodeBase64(encoded), "UTF-8");
                return mapper.readValue(decoded, ElementPair.class);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * This method
     * 1) Binary model, either compressed or not. Like well-known Google Model
     * 2) Popular CSV word2vec text format
     * 3) DL4j compressed format
     * <p>
     * Please note: Only weights will be loaded by this method.
     *
     * @param path
     * @return
     */
    public static Word2Vec readWord2VecModel(String path) {
        return readWord2VecModel(new File(path));
    }

    /**
     * This method
     * 1) Binary model, either compressed or not. Like well-known Google Model
     * 2) Popular CSV word2vec text format
     * 3) DL4j compressed format
     * <p>
     * Please note: Only weights will be loaded by this method.
     *
     * @param path  path to model file
     * @param extendedModel  if TRUE, we'll try to load HS states & Huffman tree info, if FALSE, only weights will be loaded
     * @return
     */
    public static Word2Vec readWord2VecModel(String path, boolean extendedModel) {
        return readWord2VecModel(new File(path), extendedModel);
    }

    /**
     * This method
     * 1) Binary model, either compressed or not. Like well-known Google Model
     * 2) Popular CSV word2vec text format
     * 3) DL4j compressed format
     * <p>
     * Please note: Only weights will be loaded by this method.
     *
     * @param file
     * @return
     */
    public static Word2Vec readWord2VecModel(File file) {
        return readWord2VecModel(file, false);
    }

    /**
     * This method
     * 1) Binary model, either compressed or not. Like well-known Google Model
     * 2) Popular CSV word2vec text format
     * 3) DL4j compressed format
     * <p>
     * Please note: if extended data isn't available, only weights will be loaded instead.
     *
     * @param file  model file
     * @param extendedModel  if TRUE, we'll try to load HS states & Huffman tree info, if FALSE, only weights will be loaded
     * @return word2vec model
     */
    public static Word2Vec readWord2VecModel(File file, boolean extendedModel) {
        if (!file.exists() || !file.isFile()) {
            throw new ND4JIllegalStateException("File [" + file.getAbsolutePath() + "] doesn't exist");
        }

        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();
        if (originalPeriodic) {
            Nd4j.getMemoryManager().togglePeriodicGc(false);
        }
        Nd4j.getMemoryManager().setOccasionalGcFrequency(50000);

        try {
            return readWord2Vec(file, extendedModel);
        } catch (Exception readSequenceVectors) {
            try {
                return extendedModel
                        ? readAsExtendedModel(file)
                        : readAsSimplifiedModel(file);
            } catch (Exception loadFromFileException) {
                try {
                    return readAsCsv(file);
                } catch (Exception readCsvException) {
                    try {
                        return readAsBinary(file);
                    } catch (Exception readBinaryException) {
                        try {
                            return readAsBinaryNoLineBreaks(file);
                        } catch (Exception readModelException) {
                            log.error("Unable to guess input file format", readModelException);
                            throw new RuntimeException("Unable to guess input file format. Please use corresponding loader directly");
                        }
                    }
                }
            }
        }
    }

    public static Word2Vec readAsBinaryNoLineBreaks(@NonNull File file) {
        try (InputStream inputStream = fileStream(file)) {
            return readAsBinaryNoLineBreaks(inputStream);
        } catch (IOException readCsvException) {
            throw new RuntimeException(readCsvException);
        }
    }

    public static Word2Vec readAsBinaryNoLineBreaks(@NonNull InputStream inputStream) {
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();
        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();

        // try to load without linebreaks
        try {
            if (originalPeriodic) {
                Nd4j.getMemoryManager().togglePeriodicGc(true);
            }

            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);

            return readBinaryModel(inputStream, false, false);
        } catch (Exception readModelException) {
            log.error("Cannot read binary model", readModelException);
            throw new RuntimeException("Unable to guess input file format. Please use corresponding loader directly");
        }
    }

    public static Word2Vec readAsBinary(@NonNull File file) {
        try (InputStream inputStream = fileStream(file)) {
            return readAsBinary(inputStream);
        } catch (IOException readCsvException) {
            throw new RuntimeException(readCsvException);
        }
    }

    /**
     * This method loads Word2Vec model from binary input stream.
     *
     * @param inputStream  binary input stream
     * @return Word2Vec
     */
    public static Word2Vec readAsBinary(@NonNull InputStream inputStream) {
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();
        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();

        // we fallback to trying binary model instead
        try {
            log.debug("Trying binary model restoration...");

            if (originalPeriodic) {
                Nd4j.getMemoryManager().togglePeriodicGc(true);
            }

            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);

            return readBinaryModel(inputStream, true, false);
        } catch (Exception readModelException) {
            throw new RuntimeException(readModelException);
        }
    }

    public static Word2Vec readAsCsv(@NonNull File file) {
        try (InputStream inputStream = fileStream(file)) {
            return readAsCsv(inputStream);
        } catch (IOException readCsvException) {
            throw new RuntimeException(readCsvException);
        }
    }

    /**
     * This method loads Word2Vec model from csv file
     *
     * @param inputStream  input stream
     * @return Word2Vec model
     */
    public static Word2Vec readAsCsv(@NonNull InputStream inputStream) {
        VectorsConfiguration configuration = new VectorsConfiguration();

        // let's try to load this file as csv file
        try {
            log.debug("Trying CSV model restoration...");

            Pair<InMemoryLookupTable, VocabCache> pair = loadTxt(inputStream);
            Word2Vec.Builder builder = new Word2Vec
                    .Builder()
                    .lookupTable(pair.getFirst())
                    .useAdaGrad(false)
                    .vocabCache(pair.getSecond())
                    .layerSize(pair.getFirst().layerSize())
                    // we don't use hs here, because model is incomplete
                    .useHierarchicSoftmax(false)
                    .resetModel(false);

            TokenizerFactory factory = getTokenizerFactory(configuration);
            if (factory != null) {
                builder.tokenizerFactory(factory);
            }

            return builder.build();
        } catch (Exception ex) {
            throw new RuntimeException("Unable to load model in CSV format");
        }
    }

    /**
     * This method just loads full compressed model.
     */
    private static Word2Vec readAsExtendedModel(@NonNull File file) throws IOException {
        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();

        log.debug("Trying full model restoration...");

        if (originalPeriodic) {
            Nd4j.getMemoryManager().togglePeriodicGc(true);
        }

        Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);

        return readWord2Vec(file);
    }

    private static Word2Vec readAsSimplifiedModel(@NonNull File file) throws IOException {
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();
        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        InMemoryLookupTable<VocabWord> lookupTable = new InMemoryLookupTable<>();
        AbstractCache<VocabWord> vocabCache = new AbstractCache<>();
        Word2Vec vec;
        INDArray syn0 = null;
        VectorsConfiguration configuration = new VectorsConfiguration();

        log.debug("Trying simplified model restoration...");

        File tmpFileSyn0 = ND4JFileUtils.createTempFile("word2vec", "syn");
        tmpFileSyn0.deleteOnExit();
        File tmpFileConfig = ND4JFileUtils.createTempFile("word2vec", "config");
        tmpFileConfig.deleteOnExit();
        // we don't need full model, so we go directly to syn0 file

        ZipFile zipFile = new ZipFile(file);
        ZipEntry syn = zipFile.getEntry("syn0.txt");
        InputStream stream = zipFile.getInputStream(syn);

        FileUtils.copyInputStreamToFile(stream, tmpFileSyn0);

        // now we're restoring configuration saved earlier
        ZipEntry config = zipFile.getEntry("config.json");
        if (config != null) {
            stream = zipFile.getInputStream(config);

            StringBuilder builder = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    builder.append(line);
                }
            }

            configuration = VectorsConfiguration.fromJson(builder.toString().trim());
        }

        ZipEntry ve = zipFile.getEntry("frequencies.txt");
        if (ve != null) {
            stream = zipFile.getInputStream(ve);
            AtomicInteger cnt = new AtomicInteger(0);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    String[] split = line.split(" ");
                    VocabWord word = new VocabWord(Double.valueOf(split[1]), ReadHelper.decodeB64(split[0]));
                    word.setIndex(cnt.getAndIncrement());
                    word.incrementSequencesCount(Long.valueOf(split[2]));

                    vocabCache.addToken(word);
                    vocabCache.addWordToIndex(word.getIndex(), word.getLabel());

                    Nd4j.getMemoryManager().invokeGcOccasionally();
                }
            }
        }

        List<INDArray> rows = new ArrayList<>();
        // basically read up everything, call vstacl and then return model
        try (Reader reader = new CSVReader(tmpFileSyn0)) {
            AtomicInteger cnt = new AtomicInteger(0);
            while (reader.hasNext()) {
                Pair<VocabWord, float[]> pair = reader.next();
                VocabWord word = pair.getFirst();
                INDArray vector = Nd4j.create(pair.getSecond());

                if (ve != null) {
                    if (syn0 == null)
                        syn0 = Nd4j.create(vocabCache.numWords(), vector.length());

                    syn0.getRow(cnt.getAndIncrement()).assign(vector);
                } else {
                    rows.add(vector);

                    vocabCache.addToken(word);
                    vocabCache.addWordToIndex(word.getIndex(), word.getLabel());
                }

                Nd4j.getMemoryManager().invokeGcOccasionally();
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            if (originalPeriodic)
                Nd4j.getMemoryManager().togglePeriodicGc(true);

            Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);
            try {
                if (tmpFileSyn0 != null) {
                    tmpFileSyn0.delete();
                }

                if (tmpFileConfig != null) {
                    tmpFileConfig.delete();
                }
            } catch (Exception e) {
            }    //Ignore
        }

        if (syn0 == null && vocabCache.numWords() > 0)
            syn0 = Nd4j.vstack(rows);

        if (syn0 == null) {
            log.error("Can't build syn0 table");
            throw new DL4JInvalidInputException("Can't build syn0 table");
        }

        lookupTable = new InMemoryLookupTable.Builder<VocabWord>().cache(vocabCache)
                .vectorLength(syn0.columns()).useHierarchicSoftmax(false)
                .useAdaGrad(false).build();

        lookupTable.setSyn0(syn0);

        Word2Vec.Builder builder = new Word2Vec.Builder(configuration).lookupTable(lookupTable).useAdaGrad(false)
                .vocabCache(vocabCache).layerSize(lookupTable.layerSize())
                // we don't use hs here, because model is incomplete
                .useHierarchicSoftmax(false).resetModel(false);

        /*
            Trying to restore TokenizerFactory & TokenPreProcessor
         */

        TokenizerFactory factory = getTokenizerFactory(configuration);
        if (factory != null)
            builder.tokenizerFactory(factory);

        vec = builder.build();

        return vec;
    }

    protected static TokenizerFactory getTokenizerFactory(VectorsConfiguration configuration) {
        if (configuration == null) {
            return null;
        }

        String tokenizerFactoryClassName = configuration.getTokenizerFactory();
        if (StringUtils.isNotEmpty(tokenizerFactoryClassName)) {
            TokenizerFactory factory = DL4JClassLoading.createNewInstance(tokenizerFactoryClassName);

            String tokenPreProcessorClassName = configuration.getTokenPreProcessor();
            if (StringUtils.isNotEmpty(tokenPreProcessorClassName)) {
                Object preProcessor = DL4JClassLoading.createNewInstance(tokenPreProcessorClassName);
                if(preProcessor instanceof TokenPreProcess) {
                    TokenPreProcess tokenPreProcess = (TokenPreProcess) preProcessor;
                    factory.setTokenPreProcessor(tokenPreProcess);
                }
                else {
                    log.warn("Found instance of {}, was not actually a pre processor. Ignoring.",tokenPreProcessorClassName);
                }
            }

            return factory;
        }

        return null;
    }

    /**
     * This method restores previously saved w2v model. File can be in one of the following formats:
     * 1) Binary model, either compressed or not. Like well-known Google Model
     * 2) Popular CSV word2vec text format
     * 3) DL4j compressed format
     *
     * In return you get StaticWord2Vec model, which might be used as lookup table only in multi-gpu environment.
     *
     * @param inputStream InputStream should point to previously saved w2v model
     * @return
     */
    public static WordVectors loadStaticModel(InputStream inputStream) throws IOException {

        File tmpFile = ND4JFileUtils.createTempFile("word2vec"+System.currentTimeMillis(), ".tmp");
        FileUtils.copyInputStreamToFile(inputStream, tmpFile);
        try {
            return loadStaticModel(tmpFile);
        } finally {
            tmpFile.delete();
        }

    }

    // TODO: this method needs better name :)
    /**
     * This method restores previously saved w2v model. File can be in one of the following formats:
     * 1) Binary model, either compressed or not. Like well-known Google Model
     * 2) Popular CSV word2vec text format
     * 3) DL4j compressed format
     *
     * In return you get StaticWord2Vec model, which might be used as lookup table only in multi-gpu environment.
     *
     * @param file File
     * @return
     */
    public static WordVectors loadStaticModel(@NonNull File file) {
        if (!file.exists() || file.isDirectory())
            throw new RuntimeException(
                    new FileNotFoundException("File [" + file.getAbsolutePath() + "] was not found"));

        int originalFreq = Nd4j.getMemoryManager().getOccasionalGcFrequency();
        boolean originalPeriodic = Nd4j.getMemoryManager().isPeriodicGcActive();

        if (originalPeriodic)
            Nd4j.getMemoryManager().togglePeriodicGc(false);

        Nd4j.getMemoryManager().setOccasionalGcFrequency(50000);

        CompressedRamStorage<Integer> storage = new CompressedRamStorage.Builder<Integer>().useInplaceCompression(false)
                .setCompressor(new NoOp()).emulateIsAbsent(false).build();

        VocabCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();


        // now we need to define which file format we have here
        // if zip - that's dl4j format
        try {
            log.debug("Trying DL4j format...");
            File tmpFileSyn0 = ND4JFileUtils.createTempFile("word2vec", "syn");
            tmpFileSyn0.deleteOnExit();

            ZipFile zipFile = new ZipFile(file);
            ZipEntry syn0 = zipFile.getEntry("syn0.txt");
            InputStream stream = zipFile.getInputStream(syn0);

            FileUtils.copyInputStreamToFile(stream, tmpFileSyn0);
            storage.clear();

            try (Reader reader = new CSVReader(tmpFileSyn0)) {
                while (reader.hasNext()) {
                    Pair<VocabWord, float[]> pair = reader.next();
                    VocabWord word = pair.getFirst();
                    storage.store(word.getIndex(), pair.getSecond());

                    vocabCache.addToken(word);
                    vocabCache.addWordToIndex(word.getIndex(), word.getLabel());

                    Nd4j.getMemoryManager().invokeGcOccasionally();
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                if (originalPeriodic)
                    Nd4j.getMemoryManager().togglePeriodicGc(true);

                Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);

                try {
                    tmpFileSyn0.delete();
                } catch (Exception e) {
                    //
                }
            }
        } catch (Exception e) {
            //
            try {
                // try to load file as text csv
                vocabCache = new AbstractCache.Builder<VocabWord>().build();
                storage.clear();
                log.debug("Trying CSVReader...");
                try (Reader reader = new CSVReader(file)) {
                    while (reader.hasNext()) {
                        Pair<VocabWord, float[]> pair = reader.next();
                        VocabWord word = pair.getFirst();
                        storage.store(word.getIndex(), pair.getSecond());

                        vocabCache.addToken(word);
                        vocabCache.addWordToIndex(word.getIndex(), word.getLabel());

                        Nd4j.getMemoryManager().invokeGcOccasionally();
                    }
                } catch (Exception ef) {
                    // we throw away this exception, and trying to load data as binary model
                    throw new RuntimeException(ef);
                } finally {
                    if (originalPeriodic)
                        Nd4j.getMemoryManager().togglePeriodicGc(true);

                    Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);
                }
            } catch (Exception ex) {
                // otherwise it's probably google model. which might be compressed or not
                log.debug("Trying BinaryReader...");
                vocabCache = new AbstractCache.Builder<VocabWord>().build();
                storage.clear();
                try (Reader reader = new BinaryReader(file)) {
                    while (reader.hasNext()) {
                        Pair<VocabWord, float[]> pair = reader.next();
                        VocabWord word = pair.getFirst();

                        storage.store(word.getIndex(), pair.getSecond());

                        vocabCache.addToken(word);
                        vocabCache.addWordToIndex(word.getIndex(), word.getLabel());

                        Nd4j.getMemoryManager().invokeGcOccasionally();
                    }
                } catch (Exception ez) {
                    throw new RuntimeException("Unable to guess input file format");
                } finally {
                    if (originalPeriodic)
                        Nd4j.getMemoryManager().togglePeriodicGc(true);

                    Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);
                }
            } finally {
                if (originalPeriodic)
                    Nd4j.getMemoryManager().togglePeriodicGc(true);

                Nd4j.getMemoryManager().setOccasionalGcFrequency(originalFreq);
            }
        }


        StaticWord2Vec word2Vec = new StaticWord2Vec.Builder(storage, vocabCache).build();

        return word2Vec;
    }


    protected interface Reader extends AutoCloseable {
        boolean hasNext();

        Pair<VocabWord, float[]> next();
    }


    protected static class BinaryReader implements Reader {

        protected DataInputStream stream;
        protected String nextWord;
        protected int numWords;
        protected int vectorLength;
        protected AtomicInteger idxCounter = new AtomicInteger(0);


        protected BinaryReader(@NonNull File file) {
            try {
                // Try to read as GZip
                stream = new DataInputStream(new BufferedInputStream(new GZIPInputStream(new FileInputStream(file))));
            }
            catch (IOException e) {
                try {
                    // Failed to read as Gzip, assuming it's not compressed binary format
                    stream = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
                } catch (Exception e1) {
                    throw new RuntimeException(e1);
                }
            }
            catch (Exception e) {
                throw new RuntimeException(e);
            }
            try {
                numWords = Integer.parseInt(ReadHelper.readString(stream));
                vectorLength = Integer.parseInt(ReadHelper.readString(stream));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public boolean hasNext() {
            return idxCounter.get() < numWords;
        }

        @Override
        public Pair<VocabWord, float[]> next() {
            try {
                String word = ReadHelper.readString(stream);
                VocabWord element = new VocabWord(1.0, word);
                element.setIndex(idxCounter.getAndIncrement());

                float[] vector = new float[vectorLength];
                for (int i = 0; i < vectorLength; i++) {
                    vector[i] = ReadHelper.readFloat(stream);
                }

                return Pair.makePair(element, vector);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @Override
        public void close() throws Exception {
            if (stream != null)
                stream.close();
        }
    }

    protected static class CSVReader implements Reader {
        private BufferedReader reader;
        private AtomicInteger idxCounter = new AtomicInteger(0);
        private String nextLine;

        protected CSVReader(@NonNull File file) {
            try {
                reader = new BufferedReader(new FileReader(file));
                nextLine = reader.readLine();

                // checking if there's header inside
                String[] split = nextLine.split(" ");
                try {
                    if (Integer.parseInt(split[0]) > 0 && split.length <= 5) {
                        // this is header. skip it.
                        nextLine = reader.readLine();
                    }
                } catch (Exception e) {
                    // this is proper string, do nothing
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        public boolean hasNext() {
            return nextLine != null;
        }

        public Pair<VocabWord, float[]> next() {

            String[] split = nextLine.split(" ");

            VocabWord word = new VocabWord(1.0, ReadHelper.decodeB64(split[0]));
            word.setIndex(idxCounter.getAndIncrement());

            float[] vector = new float[split.length - 1];
            for (int i = 1; i < split.length; i++) {
                vector[i - 1] = Float.parseFloat(split[i]);
            }

            try {
                nextLine = reader.readLine();
            } catch (Exception e) {
                nextLine = null;
            }

            return Pair.makePair(word, vector);
        }

        @Override
        public void close() throws Exception {
            if (reader != null)
                reader.close();
        }
    }

    /**
     * This method saves Word2Vec model to output stream
     *
     * @param word2Vec Word2Vec
     * @param stream OutputStream
     */
    public static void writeWord2Vec(@NonNull Word2Vec word2Vec, @NonNull OutputStream stream)
            throws IOException {

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(word2Vec.getConfiguration())
                .layerSize(word2Vec.getLayerSize()).build();
        vectors.setVocab(word2Vec.getVocab());
        vectors.setLookupTable(word2Vec.getLookupTable());
        vectors.setModelUtils(word2Vec.getModelUtils());
        writeSequenceVectors(vectors, stream);
    }

    /**
     * This method restores Word2Vec model from file
     *
     * @param path
     * @param readExtendedTables
     * @return Word2Vec
     */
    public static Word2Vec readWord2Vec(@NonNull String path, boolean readExtendedTables) {
        File file = new File(path);
        return readWord2Vec(file, readExtendedTables);
    }

    /**
     * This method saves table of weights to file
     *
     * @param weightLookupTable WeightLookupTable
     * @param file File
     */
    public static <T extends SequenceElement>  void writeLookupTableBinary(WeightLookupTable<T> weightLookupTable,
                                                                           @NonNull File file) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file),
                StandardCharsets.UTF_8))) {
            int numWords = weightLookupTable.getVocabCache().numWords();
            int layersSize = weightLookupTable.layerSize();
            long totalNumberOfDocs = weightLookupTable.getVocabCache().totalNumberOfDocs();

            String format = "%d %d %d\n";
            String header = String.format(format, numWords, layersSize, totalNumberOfDocs);

            writer.write(header);

            String row = "";
            for (int j = 0; j < weightLookupTable.getVocabCache().words().size(); ++j) {
                String label =  weightLookupTable.getVocabCache().wordAtIndex(j);
                row += label + " ";
                int freq = weightLookupTable.getVocabCache().wordFrequency(label);
                int rows = ((InMemoryLookupTable)weightLookupTable).getSyn0().rows();
                int cols = ((InMemoryLookupTable)weightLookupTable).getSyn0().columns();
                row += freq + " " + rows + " " + cols + " ";

                for (int r = 0; r < rows; ++r) {
                    //row += " ";
                    for (int c = 0; c < cols; ++c) {
                        row += ((InMemoryLookupTable) weightLookupTable).getSyn0().getDouble(r, c) + " ";
                    }
                    //row += " ";
                }
                row += "\n";
            }
            writer.write(row);
        }
    }

    public static <T extends SequenceElement> WeightLookupTable<T> readLookupTable(File file)
            throws IOException {
        return readLookupTable(new FileInputStream(file));
    }

    public static <T extends SequenceElement> WeightLookupTable<T> readLookupTable(InputStream stream)
            throws IOException {
        WeightLookupTable<T> weightLookupTable = null;
        AbstractCache<VocabWord> vocabCache = new AbstractCache<>();
        final int startSyn0 = 4;
        boolean headerRead = false;
        int numWords = -1, layerSize = -1, totalNumberOfDocs = -1;
        try {
            INDArray syn0 = null;
            int index = 0;
            for (String line : IOUtils.readLines(stream)) {
                String[] tokens = line.split(" ");
                if (!headerRead) {
                    // reading header as "NUM_WORDS VECTOR_SIZE NUM_DOCS"
                    numWords = Integer.parseInt(tokens[0]);
                    layerSize = Integer.parseInt(tokens[1]);
                    totalNumberOfDocs = Integer.parseInt(tokens[2]);
                    log.debug("Reading header - words: {}, layerSize: {}, totalNumberOfDocs: {}",
                            numWords, layerSize, totalNumberOfDocs);
                    headerRead = true;
                    weightLookupTable = new InMemoryLookupTable.Builder().cache(vocabCache).vectorLength(layerSize).build();
                } else {
                    String label = ReadHelper.decodeB64(tokens[0]);
                    int freq = Integer.parseInt(tokens[1]);
                    int rows = Integer.parseInt(tokens[2]);
                    int cols = Integer.parseInt(tokens[3]);

                    if (syn0 == null)
                        syn0  = Nd4j.createUninitialized(rows, cols);

                    int i = startSyn0;
                    for (int r = 0; r < rows; ++r) {
                        double[] vector = new double[cols];
                        for (int c = 0;  c < cols; ++c) {
                            vector[c] = Double.parseDouble(tokens[i]);
                            ++i;
                        }
                        syn0.putRow(r, Nd4j.create(vector));
                    }

                    VocabWord vw = new VocabWord(freq, label);
                    vw.setIndex(index);
                    weightLookupTable.getVocabCache().addToken((T)vw);
                    weightLookupTable.getVocabCache().addWordToIndex(index, label);
                    ++index;
                }
            }
            ((InMemoryLookupTable<T>) weightLookupTable).setSyn0(syn0);
        }
        finally {
            stream.close();
        }
        return weightLookupTable;
    }

    /**
     * This method loads Word2Vec model from file
     *
     * @param file File
     * @param readExtendedTables boolean
     * @return Word2Vec
     */
    public static Word2Vec readWord2Vec(@NonNull File file, boolean readExtendedTables) {
        try (InputStream inputStream = fileStream(file)) {
            return readWord2Vec(inputStream, readExtendedTables);
        } catch (Exception readSequenceVectors) {
            throw new RuntimeException(readSequenceVectors);
        }
    }

    /**
     * This method loads Word2Vec model from input stream
     *
     * @param stream InputStream
     * @param readExtendedTable boolean
     * @return Word2Vec
     */
    public static Word2Vec readWord2Vec(
            @NonNull InputStream stream,
            boolean readExtendedTable) throws IOException {
        SequenceVectors<VocabWord> vectors = readSequenceVectors(stream, readExtendedTable);

        Word2Vec word2Vec = new Word2Vec
                .Builder(vectors.getConfiguration())
                .layerSize(vectors.getLayerSize())
                .build();
        word2Vec.setVocab(vectors.getVocab());
        word2Vec.setLookupTable(vectors.lookupTable());
        word2Vec.setModelUtils(vectors.getModelUtils());

        return word2Vec;
    }

    /**
     * This method loads FastText model to file
     *
     * @param vectors FastText
     * @param path File
     */
    public static void writeWordVectors(@NonNull FastText vectors, @NonNull File path) throws IOException {
        ObjectOutputStream outputStream = null;
        try {
            outputStream = new ObjectOutputStream(new FileOutputStream(path ));
            outputStream.writeObject(vectors);
        }
        finally {
            try {
                if (outputStream != null) {
                    outputStream.flush();
                    outputStream.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    /**
     * This method unloads FastText model from file
     *
     * @param path File
     */
    public static FastText readWordVectors(File path) {
        FastText result = null;
        try {
            FileInputStream fileIn = new FileInputStream(path);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            try {
                result = (FastText) in.readObject();
            } catch (ClassNotFoundException ex) {

            }
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        return result;
    }

    /**
     * This method prints memory usage to log
     *
     * @param numWords
     * @param vectorLength
     * @param numTables
     */
    public static void printOutProjectedMemoryUse(long numWords, int vectorLength, int numTables) {
        double memSize = numWords * vectorLength * Nd4j.sizeOfDataType() * numTables;

        String sfx;
        double value;
        if (memSize < 1024 * 1024L) {
            sfx = "KB";
            value = memSize / 1024;
        }
        if (memSize < 1024 * 1024L * 1024L) {
            sfx = "MB";
            value = memSize / 1024 / 1024;
        } else {
            sfx = "GB";
            value = memSize / 1024 / 1024 / 1024;
        }

        OneTimeLogger.info(log, "Projected memory use for model: [{} {}]", String.format("%.2f", value), sfx);

    }

    /**
     *   Helper static methods to read data from input stream.
     */
    public static class ReadHelper {
        /**
         * Read a float from a data input stream Credit to:
         * https://github.com/NLPchina/Word2VEC_java/blob/master/src/com/ansj/vec/Word2VEC.java
         *
         * @param is
         * @return
         * @throws IOException
         */
        private static float readFloat(InputStream is) throws IOException {
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
        private static float getFloat(byte[] b) {
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
                    sb.append(new String(bytes, "UTF-8"));
                    i = -1;
                    bytes = new byte[MAX_SIZE];
                }
            }
            sb.append(new String(bytes, 0, i + 1, "UTF-8"));
            return sb.toString();
        }

        private static final String B64 = "B64:";

        /**
         * Encode input string
         *
         * @param word String
         * @return String
         */
        public static String encodeB64(String word) {
            try {
                return B64 + Base64.encodeBase64String(word.getBytes("UTF-8")).replaceAll("(\r|\n)", "");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        /**
         * Encode input string
         *
         * @param word String
         * @return String
         */

        public static String decodeB64(String word) {
            if (word.startsWith(B64)) {
                String arp = word.replaceFirst(B64, "");
                try {
                    return new String(Base64.decodeBase64(arp), "UTF-8");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else
                return word;
        }
    }
}
