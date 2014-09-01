package org.deeplearning4j.word2vec.wordstore.ehcache;

import net.sf.ehcache.Cache;
import net.sf.ehcache.CacheManager;
import net.sf.ehcache.Element;
import net.sf.ehcache.search.Query;
import net.sf.ehcache.search.Result;
import net.sf.ehcache.search.Results;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.word2vec.VocabWord;
import org.deeplearning4j.word2vec.wordstore.VocabCache;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

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

    private CacheManager cacheManager;
    private Cache cache;

    public EhCacheVocabCache() {
        ClassPathResource resource = new ClassPathResource("ehcache.xml");
        try {
            cacheManager = CacheManager.create(resource.getURL());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        cache = cacheManager.getCache(CACHE_NAME);

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
    }

    @Override
    public int wordFrequency(String word) {
        Element element = cache.get(word + "-"  + FREQUENCY);
        if(element != null && element.getObjectValue() != null)
            return (Integer) element.getObjectValue();
        return 0;
    }

    @Override
    public boolean containsWord(String word) {
        return retrieve(word + "-" +  VOCAB ) != null;
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
        return NDArrays.create(vectors,new int[]{codes.length,vectors.get(0).columns()});
    }

    @Override
    public Collection<VocabWord> vocabWords() {
        Results r = cache.createQuery().addCriteria(Query.KEY.ilike(VOCAB)).execute();
        List<Result> results = r.all();
        List<VocabWord> ret = new ArrayList<>();
        for(Result r2 : results) {
            ret.add((VocabWord) r2.getValue());
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
        store(index,word);
        incrementWordCount(word);
        incrementWordCountBy(1);
        store(word + "-" +  VOCAB ,word);
    }

    @Override
    public void putVocabWord(String word, VocabWord vocabWord) {
           store(word + "-" + VOCAB,vocabWord);

    }

    @Override
    public int numWords() {
        Integer numWords = retrieve(NUM_WORDS);
        if(numWords == null)
            return 0;
        return numWords;
    }



    private void store(Object key,Object value) {
        Element e = new Element(key,value);
        cache.put(e);
    }


    private void incrementWordCountBy(int num) {
        Integer curr = retrieve(NUM_WORDS);
        if(curr == null)
            curr = 0;
        store(NUM_WORDS ,curr + num);
    }


    private <E> E retrieve(Object key) {
        Element e = cache.get(key);
        if(e != null)
            return (E) cache.get(key).getObjectValue();
        return null;
    }

}
