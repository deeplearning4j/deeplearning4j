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

package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.nd4j.shade.jackson.annotation.JsonAutoDetect;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.nd4j.shade.jackson.databind.*;
import org.nd4j.shade.jackson.databind.type.CollectionType;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * This is generic VocabCache implementation designed as abstract SequenceElements vocabulary
 *
 * @author raver119@gmail.com
 * @author alexander@skymind.io
 */
@Slf4j
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
@JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY, getterVisibility = JsonAutoDetect.Visibility.NONE,
        setterVisibility = JsonAutoDetect.Visibility.NONE)
public class AbstractCache<T extends SequenceElement> implements VocabCache<T> {
    private static final String CLASS_FIELD = "@class";
    private static final String VOCAB_LIST_FIELD = "VocabList";
    private static final String VOCAB_ITEM_FIELD = "VocabItem";
    private static final String DOC_CNT_FIELD = "DocumentsCounter";
    private static final String MINW_FREQ_FIELD = "MinWordsFrequency";
    private static final String HUGE_MODEL_FIELD = "HugeModelExpected";
    private static final String STOP_WORDS_FIELD = "StopWords";
    private static final String SCAVENGER_FIELD = "ScavengerThreshold";
    private static final String RETENTION_FIELD = "RetentionDelay";
    private static final String TOTAL_WORD_FIELD = "TotalWordCount";

    private final ConcurrentMap<Long, T> vocabulary = new ConcurrentHashMap<>();

    private final Map<String, T> extendedVocabulary = new ConcurrentHashMap<>();

    private final Map<Integer, T> idxMap = new ConcurrentHashMap<>();

    private final AtomicLong documentsCounter = new AtomicLong(0);

    private int minWordFrequency = 0;
    private boolean hugeModelExpected = false;

    // we're using <String>for compatibility & failproof reasons: it's easier to store unique labels then abstract objects of unknown size
    // TODO: wtf this one is doing here?
    private List<String> stopWords = new ArrayList<>(); // stop words

    // this variable defines how often scavenger will be activated
    private int scavengerThreshold = 3000000; // ser
    private int retentionDelay = 3; // ser

    // for scavenger mechanics we need to know the actual number of words being added
    private transient AtomicLong hiddenWordsCounter = new AtomicLong(0);

    private final AtomicLong totalWordCount = new AtomicLong(0); // ser

    private static final int MAX_CODE_LENGTH = 40;

    /**
     * Deserialize vocabulary from specified path
     */
    @Override
    public void loadVocab() {
        // TODO: this method should be static and accept path
    }

    /**
     * Returns true, if number of elements in vocabulary > 0, false otherwise
     *
     * @return
     */
    @Override
    public boolean vocabExists() {
        return !vocabulary.isEmpty();
    }

    /**
     * Serialize vocabulary to specified path
     *
     */
    @Override
    public void saveVocab() {
        // TODO: this method should be static and accept path
    }

    /**
     * Returns collection of labels available in this vocabulary
     *
     * @return
     */
    @Override
    public Collection<String> words() {
        return Collections.unmodifiableCollection(extendedVocabulary.keySet());
    }

    /**
     * Increment frequency for specified label by 1
     *
     * @param word the word to increment the count for
     */
    @Override
    public void incrementWordCount(String word) {
        incrementWordCount(word, 1);
    }


    /**
     * Increment frequency for specified label by specified value
     *
     * @param word the word to increment the count for
     * @param increment the amount to increment by
     */
    @Override
    public void incrementWordCount(String word, int increment) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            element.increaseElementFrequency(increment);
            totalWordCount.addAndGet(increment);
        }
    }

    /**
     * Returns the SequenceElement's frequency over training corpus
     *
     * @param word the word to retrieve the occurrence frequency for
     * @return
     */
    @Override
    public int wordFrequency(@NonNull String word) {
        // TODO: proper wordFrequency impl should return long, instead of int
        T element = extendedVocabulary.get(word);
        if (element != null)
            return (int) element.getElementFrequency();
        return 0;
    }

    /**
     * Checks, if specified label exists in vocabulary
     *
     * @param word the word to check for
     * @return
     */
    @Override
    public boolean containsWord(String word) {
        return extendedVocabulary.containsKey(word);
    }

    /**
     * Checks, if specified element exists in vocabulary
     *
     * @param element
     * @return
     */
    public boolean containsElement(T element) {
        // FIXME: lolwtf
        return vocabulary.values().contains(element);
    }

    /**
     * Returns the label of the element at specified Huffman index
     *
     * @param index the index of the word to get
     * @return
     */
    @Override
    public String wordAtIndex(int index) {
        T element = idxMap.get(index);
        if (element != null) {
            return element.getLabel();
        }
        return null;
    }

    /**
     * Returns SequenceElement at specified index
     *
     * @param index
     * @return
     */
    @Override
    public T elementAtIndex(int index) {
        return idxMap.get(index);
    }

    /**
     * Returns Huffman index for specified label
     *
     * @param label the label to get index for
     * @return >=0 if label exists, -1 if Huffman tree wasn't built yet, -2 if specified label wasn't found
     */
    @Override
    public int indexOf(String label) {
        T token = tokenFor(label);
        if (token != null) {
            return token.getIndex();
        } else
            return -2;
    }

    /**
     * Returns collection of SequenceElements stored in this vocabulary
     *
     * @return
     */
    @Override
    public Collection<T> vocabWords() {
        return vocabulary.values();
    }

    /**
     * Returns total number of elements observed
     *
     * @return
     */
    @Override
    public long totalWordOccurrences() {
        return totalWordCount.get();
    }

    public void setTotalWordOccurences(long value) {
        totalWordCount.set(value);
    }

    /**
     * Returns SequenceElement for specified label
     *
     * @param label to fetch element for
     * @return
     */
    @Override
    public T wordFor(@NonNull String label) {
        return extendedVocabulary.get(label);
    }

    @Override
    public T wordFor(long id) {
        return vocabulary.get(id);
    }

    /**
     * This method allows to insert specified label to specified Huffman tree position.
     * CAUTION: Never use this, unless you 100% sure what are you doing.
     *
     * @param index
     * @param label
     */
    @Override
    public void addWordToIndex(int index, String label) {
        if (index >= 0) {
            T token = tokenFor(label);
            if (token != null) {
                idxMap.put(index, token);
                token.setIndex(index);
            }
        }
    }

    @Override
    public void addWordToIndex(int index, long elementId) {
        if (index >= 0)
            idxMap.put(index, tokenFor(elementId));
    }

    @Override
    @Deprecated
    public void putVocabWord(String word) {
        if (!containsWord(word))
            throw new IllegalStateException("Specified label is not present in vocabulary");
    }

    /**
     * Returns number of elements in this vocabulary
     *
     * @return
     */
    @Override
    public int numWords() {
        return vocabulary.size();
    }

    /**
     * Returns number of documents (if applicable) the label was observed in.
     *
     * @param word the number of documents the word appeared in
     * @return
     */
    @Override
    public int docAppearedIn(String word) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            return (int) element.getSequencesCount();
        } else
            return -1;
    }

    /**
     * Increment number of documents the label was observed in
     *
     * Please note: this method is NOT thread-safe
     *
     * @param word the word to increment by
     * @param howMuch
     */
    @Override
    public void incrementDocCount(String word, long howMuch) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            element.incrementSequencesCount();
        }
    }

    /**
     * Set exact number of observed documents that contain specified word
     *
     * Please note: this method is NOT thread-safe
     *
     * @param word the word to set the count for
     * @param count the count of the word
     */
    @Override
    public void setCountForDoc(String word, long count) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            element.setSequencesCount(count);
        }
    }

    /**
     * Returns total number of documents observed (if applicable)
     *
     * @return
     */
    @Override
    public long totalNumberOfDocs() {
        return documentsCounter.intValue();
    }

    /**
     * Increment total number of documents observed by 1
     */
    @Override
    public void incrementTotalDocCount() {
        documentsCounter.incrementAndGet();
    }

    /**
     * Increment total number of documents observed by specified value
     */
    @Override
    public void incrementTotalDocCount(long by) {
        documentsCounter.addAndGet(by);
    }

    /**
     * This method allows to set total number of documents
     * @param by
     */
    public void setTotalDocCount(long by) {

        documentsCounter.set(by);
    }


    /**
     * Returns collection of SequenceElements from this vocabulary. The same as vocabWords() method
     *
     * @return collection of SequenceElements
     */
    @Override
    public Collection<T> tokens() {
        return vocabWords();
    }

    /**
     * This method adds specified SequenceElement to vocabulary
     *
     * @param element the word to add
     */
    @Override
    public boolean addToken(T element) {
        boolean ret = false;
        T oldElement = vocabulary.putIfAbsent(element.getStorageId(), element);
        if (oldElement == null) {
            //putIfAbsent added our element
            if (element.getLabel() != null) {
                extendedVocabulary.put(element.getLabel(), element);
            }
            oldElement = element;
            ret = true;
        } else {
            oldElement.incrementSequencesCount(element.getSequencesCount());
            oldElement.increaseElementFrequency((int) element.getElementFrequency());
        }
        totalWordCount.addAndGet((long) oldElement.getElementFrequency());
        return ret;
    }

    public void addToken(T element, boolean lockf) {
        T oldElement = vocabulary.putIfAbsent(element.getStorageId(), element);
        if (oldElement == null) {
            //putIfAbsent added our element
            if (element.getLabel() != null) {
                extendedVocabulary.put(element.getLabel(), element);
            }
            oldElement = element;
        } else {
            oldElement.incrementSequencesCount(element.getSequencesCount());
            oldElement.increaseElementFrequency((int) element.getElementFrequency());
        }
        totalWordCount.addAndGet((long) oldElement.getElementFrequency());
    }

    /**
     * Returns SequenceElement for specified label. The same as wordFor() method.
     *
     * @param label the label to get the token for
     * @return
     */
    @Override
    public T tokenFor(String label) {
        return wordFor(label);
    }

    @Override
    public T tokenFor(long id) {
        return vocabulary.get(id);
    }

    /**
     * Checks, if specified label already exists in vocabulary. The same as containsWord() method.
     *
     * @param label the token to test
     * @return
     */
    @Override
    public boolean hasToken(String label) {
        return containsWord(label);
    }


    /**
     * This method imports all elements from VocabCache passed as argument
     * If element already exists,
     *
     * @param vocabCache
     */
    public void importVocabulary(@NonNull VocabCache<T> vocabCache) {
        AtomicBoolean added = new AtomicBoolean(false);
        for (T element : vocabCache.vocabWords()) {
            if (this.addToken(element))
                added.set(true);
        }
        //logger.info("Current state: {}; Adding value: {}", this.documentsCounter.get(), vocabCache.totalNumberOfDocs());
        if (added.get())
            this.documentsCounter.addAndGet(vocabCache.totalNumberOfDocs());
    }

    @Override
    public void updateWordsOccurrences() {
        totalWordCount.set(0);
        for (T element : vocabulary.values()) {
            long value = (long) element.getElementFrequency();

            if (value > 0) {
                totalWordCount.addAndGet(value);
            }
        }
        log.info("Updated counter: [" + totalWordCount.get() + "]");
    }

    @Override
    public void removeElement(String label) {
        SequenceElement element = extendedVocabulary.get(label);
        if (element != null) {
            totalWordCount.getAndAdd((long) element.getElementFrequency() * -1);
            idxMap.remove(element.getIndex());
            extendedVocabulary.remove(label);
            vocabulary.remove(element.getStorageId());
        } else
            throw new IllegalStateException("Can't get label: '" + label + "'");
    }

    @Override
    public void removeElement(T element) {
        removeElement(element.getLabel());
    }

    private static ObjectMapper mapper = null;
    private static final Object lock = new Object();

    private static ObjectMapper mapper() {
        if (mapper == null) {
            synchronized (lock) {
                if (mapper == null) {
                    mapper = new ObjectMapper();
                    mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
                    mapper.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
                    return mapper;
                }
            }
        }
        return mapper;
    }

    public String toJson() throws JsonProcessingException {

        JsonObject retVal = new JsonObject();
        ObjectMapper mapper = mapper();
        Iterator<T> iter = vocabulary.values().iterator();
        Class clazz = null;
        if (iter.hasNext())
            clazz = iter.next().getClass();
        else
            return retVal.getAsString();

        retVal.addProperty(CLASS_FIELD, mapper.writeValueAsString(this.getClass().getName()));

        JsonArray jsonValues = new JsonArray();
        for (T value : vocabulary.values()) {
            JsonObject item = new JsonObject();
            item.addProperty(CLASS_FIELD, mapper.writeValueAsString(clazz));
            item.addProperty(VOCAB_ITEM_FIELD, mapper.writeValueAsString(value));
            jsonValues.add(item);
        }
        retVal.add(VOCAB_LIST_FIELD, jsonValues);

        retVal.addProperty(DOC_CNT_FIELD, mapper.writeValueAsString(documentsCounter.longValue()));
        retVal.addProperty(MINW_FREQ_FIELD, mapper.writeValueAsString(minWordFrequency));
        retVal.addProperty(HUGE_MODEL_FIELD, mapper.writeValueAsString(hugeModelExpected));

        retVal.addProperty(STOP_WORDS_FIELD, mapper.writeValueAsString(stopWords));

        retVal.addProperty(SCAVENGER_FIELD, mapper.writeValueAsString(scavengerThreshold));
        retVal.addProperty(RETENTION_FIELD, mapper.writeValueAsString(retentionDelay));
        retVal.addProperty(TOTAL_WORD_FIELD, mapper.writeValueAsString(totalWordCount.longValue()));

        return retVal.toString();
    }

    public static <T extends SequenceElement> AbstractCache<T> fromJson(String jsonString)  throws IOException {
        AbstractCache<T> retVal = new AbstractCache.Builder<T>().build();

        JsonParser parser = new JsonParser();
        JsonObject json = parser.parse(jsonString).getAsJsonObject();

        ObjectMapper mapper = mapper();

        CollectionType wordsCollectionType = mapper.getTypeFactory()
                .constructCollectionType(List.class, VocabWord.class);

        List<T> items = new ArrayList<>();
        JsonArray jsonArray = json.get(VOCAB_LIST_FIELD).getAsJsonArray();
        for (int i = 0; i < jsonArray.size(); ++i) {
            VocabWord item = mapper.readValue(jsonArray.get(i).getAsJsonObject().get(VOCAB_ITEM_FIELD).getAsString(), VocabWord.class);
            items.add((T)item);
        }

        ConcurrentMap<Long, T> vocabulary = new ConcurrentHashMap<>();
        Map<String, T> extendedVocabulary = new ConcurrentHashMap<>();
        Map<Integer, T> idxMap = new ConcurrentHashMap<>();

        for (T item : items) {
            vocabulary.put(item.getStorageId(), item);
            extendedVocabulary.put(item.getLabel(), item);
            idxMap.put(item.getIndex(), item);
        }
        List<String> stopWords = mapper.readValue(json.get(STOP_WORDS_FIELD).getAsString(), List.class);

        Long documentsCounter = json.get(DOC_CNT_FIELD).getAsLong();
        Integer minWordsFrequency = json.get(MINW_FREQ_FIELD).getAsInt();
        Boolean hugeModelExpected = json.get(HUGE_MODEL_FIELD).getAsBoolean();
        Integer scavengerThreshold = json.get(SCAVENGER_FIELD).getAsInt();
        Integer retentionDelay = json.get(RETENTION_FIELD).getAsInt();
        Long totalWordCount = json.get(TOTAL_WORD_FIELD).getAsLong();

        retVal.vocabulary.putAll(vocabulary);
        retVal.extendedVocabulary.putAll(extendedVocabulary);
        retVal.idxMap.putAll(idxMap);
        retVal.stopWords.addAll(stopWords);
        retVal.documentsCounter.set(documentsCounter);
        retVal.minWordFrequency = minWordsFrequency;
        retVal.hugeModelExpected = hugeModelExpected;
        retVal.scavengerThreshold = scavengerThreshold;
        retVal.retentionDelay = retentionDelay;
        retVal.totalWordCount.set(totalWordCount);
        return retVal;
    }

    public static class Builder<T extends SequenceElement> {
        protected int scavengerThreshold = 3000000;
        protected int retentionDelay = 3;
        protected int minElementFrequency;
        protected boolean hugeModelExpected = false;


        public Builder<T> hugeModelExpected(boolean reallyExpected) {
            this.hugeModelExpected = reallyExpected;
            return this;
        }

        public Builder<T> scavengerThreshold(int threshold) {
            this.scavengerThreshold = threshold;
            return this;
        }

        public Builder<T> scavengerRetentionDelay(int delay) {
            this.retentionDelay = delay;
            return this;
        }

        public Builder<T> minElementFrequency(int minFrequency) {
            this.minElementFrequency = minFrequency;
            return this;
        }

        public AbstractCache<T> build() {
            AbstractCache<T> cache = new AbstractCache<>();
            cache.minWordFrequency = this.minElementFrequency;
            cache.scavengerThreshold = this.scavengerThreshold;
            cache.retentionDelay = this.retentionDelay;

            return cache;
        }

    }
}
