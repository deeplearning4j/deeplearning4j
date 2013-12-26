package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning;

import java.util.Collection;
import java.util.PriorityQueue;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.Word2Vec.VocabWord;

public class Huffman {

	public Huffman(Collection<VocabWord> words,Word2Vec vec) {
		this.words = words;
		this.vec = vec;
	}


	private Collection<VocabWord> words;
	private PriorityQueue<VocabWord> heap = new PriorityQueue<VocabWord>();
	private Word2Vec vec;
	
	public void build() {
		heap.addAll(words);

		//build huffman tree
		for(int i = 0 ; i< words.size() - 1; i++) {
			VocabWord word1 = heap.poll();
			VocabWord word2 = heap.poll();
			double count = word1.getWordFrequency() + word2.getWordFrequency();
			VocabWord newWord = new VocabWord(count,vec.getLayerSize());
			
			word1.setCode(0);
			word2.setCode(1);
			
			word1.setParent(newWord);
			word2.setParent(newWord);
			
			newWord.setLeft(word1);
			newWord.setRight(word2);
			int idx = i + words.size();
			newWord.setIndex(idx);
			heap.add(newWord);
		}
	}

}
