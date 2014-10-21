package org.deeplearning4j.text.invertedindex;

import org.deeplearning4j.models.word2vec.VocabWord;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Created by agibsonccc on 10/21/14.
 */
public class DefaultInvertedIndex implements InvertedIndex {
    private Map<Integer,List<VocabWord>> docToWord = new ConcurrentHashMap<>();
    private Map<VocabWord,List<Integer>> wordToDocs = new ConcurrentHashMap<>();

    @Override
    public List<VocabWord> document(int index) {
        return docToWord.get(index);
    }

    @Override
    public List<Integer> documents(VocabWord vocabWord) {
        return wordToDocs.get(vocabWord);
    }

    @Override
    public int numDocuments() {
        return docToWord.size();
    }

    @Override
    public Collection<Integer> allDocs() {
        return docToWord.keySet();
    }


    @Override
    public void addWordToDoc(int doc, VocabWord word) {
       List<VocabWord> wordsForDoc = docToWord.get(doc);
        if(wordsForDoc == null) {
            wordsForDoc = Collections.synchronizedList(new ArrayList<VocabWord>());
            docToWord.put(doc,wordsForDoc);
        }

        wordsForDoc.add(word);
        List<Integer> docList = wordToDocs.get(word);
        if(docList != null)
            docList.add(doc);
        else {
            docList = new CopyOnWriteArrayList<>();
            docList.add(doc);
            wordToDocs.put(word,docList);
        }

    }

    @Override
    public void addWordsToDoc(int doc, List<VocabWord> words) {
      for(VocabWord word : words)
          addWordToDoc(doc,word);
    }
}
