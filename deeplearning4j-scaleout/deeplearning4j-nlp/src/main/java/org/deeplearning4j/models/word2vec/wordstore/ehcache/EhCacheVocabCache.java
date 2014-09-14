package org.deeplearning4j.models.word2vec.wordstore.ehcache;

import net.sf.ehcache.Cache;
import net.sf.ehcache.CacheManager;
import net.sf.ehcache.Element;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 * An ehcache vocab cache
 *
 *
 * @author Adam Gibson
 */
public class EhCacheVocabCache implements VocabCache {


    public final static String VOCAB = "vocab";
    public final static String COUNT = "count";
    public final static String INDEX = "index";
    public final static String VECTOR = "vector";
    public final static String CACHE_NAME = "vocabCache";
    public final static String FREQUENCY = "frequency";
    public final static String WORD_OCCURRENCES = "word_occurrences";
    public final static String NUM_WORDS = "numwords";
    public final static String CODE = "codes";
    private AtomicInteger numWords = new AtomicInteger(0);
    private CacheManager cacheManager;
    private Cache cache;

    public EhCacheVocabCache() {
        ClassPathResource resource = new ClassPathResource("org/deeplearning4j/ehcache.xml");
        try {
            cacheManager = CacheManager.create(resource.getURL());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        cache = cacheManager.getCache(CACHE_NAME);


        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {

            /**
             * When an object implementing interface <code>Runnable</code> is used
             * to create a thread, starting the thread causes the object's
             * <code>run</code> method to be called in that separately executing
             * thread.
             * <p/>
             * The general contract of the method <code>run</code> is that it may
             * take any action whatsoever.
             *
             * @see Thread#run()
             */
            @Override
            public void run() {
                cache.flush();
            }
        }));



        if(numWords.get() == 0) {
            Integer numWords = retrieve(NUM_WORDS);
            if(numWords != null)
                this.numWords.set(numWords);
        }
    }






    @Override
    public void incrementWordCount(String word) {
        incrementWordCount(word,1);

    }

    @Override
    public void incrementWordCount(String word, int increment) {
        Integer val = retrieve(word + "-" + COUNT);
        if(val != null ) {
            Integer count =  val;
            cache.put(new Element(word + "-" + COUNT,count + increment));

        }
        else {
            Element e = new Element(word + "-" + COUNT,increment);
            cache.put(e);
        }

        incrementWordCountBy(increment);
    }

    @Override
    public int wordFrequency(String word) {
        Element element = cache.get(word + "-"  + COUNT);
        if(element != null && element.getObjectValue() != null)
            return (Integer) element.getObjectValue();
        return 0;
    }

    @Override
    public boolean containsWord(String word) {
        return indexOf(word) >= 0;
    }

    @Override
    public String wordAtIndex(int index) {
        return retrieve(index);
    }

    @Override
    public int indexOf(String word) {
        Element ele = cache.get(word + "-" + INDEX);
        if(ele != null && ele.getObjectValue() != null)
            return (Integer) ele.getObjectValue();
        return -1;
    }

    @Override
    public void putCode(int codeIndex, INDArray code) {
        store(codeIndex,code);
    }

    @Override
    public INDArray loadCodes(int[] codes) {
        List<INDArray> vectors = new ArrayList<>();
        for(int i : codes) {
            INDArray a = retrieve(i + "-" + CODE);
            vectors.add(a);

        }
        return Nd4j.create(vectors,new int[]{codes.length,vectors.get(0).columns()});
    }

    @Override
    public Collection<VocabWord> vocabWords() {
        List<Object> keys = cache.getKeys();
        List<VocabWord> ret = new ArrayList<>();
        for(Object key  : keys) {
            if(key.toString().contains("-" + VOCAB)) {
                Object val = cache.get(key).getObjectValue();
                if(val instanceof VocabWord) {
                    VocabWord w = (VocabWord) val;
                    ret.add(w);
                }


            }
        }

        return ret;
    }

    @Override
    public int totalWordOccurrences() {
        return retrieve(WORD_OCCURRENCES);
    }

    @Override
    public void putVector(String word, INDArray vector) {
        store(word + "-" + VECTOR,vector);
    }

    @Override
    public INDArray vector(String word) {
        INDArray vector = retrieve(word + "-" + VECTOR);
        return vector;
    }

    @Override
    public VocabWord wordFor(String word) {
        return retrieve(word + "-" + VOCAB);
    }

    @Override
    public void addWordToIndex(int index, String word) {
        assert indexOf(word) < 0 : "Word should not already be in index.";
        store(index,word);
        store(word + "-" + INDEX,index);
        //this word actually belongs in the vocab
        incrementWordCountBy(1);
    }

    @Override
    public void putVocabWord(String word, VocabWord vocabWord) {
        store(word + "-" + VOCAB,vocabWord);

    }

    @Override
    public synchronized  int numWords() {
        return numWords.get();
    }



    private void store(Object key,Object value) {
        assert key != null && value != null : "Unable to store null values";
        Element e = new Element(key,value);
        cache.put(e);
    }


    private void incrementWordCountBy(int num) {
        numWords.set(numWords.get() + num);
        store(NUM_WORDS,numWords.get() + num);
    }


    private <E> E retrieve(Object key) {
        Element e = cache.get(key);
        if(e != null)
            return (E) cache.get(key).getObjectValue();
        return null;
    }

}
