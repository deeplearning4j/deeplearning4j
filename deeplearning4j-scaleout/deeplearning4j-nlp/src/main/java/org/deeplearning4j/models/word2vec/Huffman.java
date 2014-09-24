package org.deeplearning4j.models.word2vec;

import java.util.Collection;
import java.util.PriorityQueue;
import java.util.Stack;

import org.deeplearning4j.berkeley.Triple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Huffman tree builder
 * @author Adam Gibson
 *
 */
public class Huffman {

	public Huffman(Collection<VocabWord> words) {
		this.words = words;
	}


	private static Logger log = LoggerFactory.getLogger(Huffman.class);

	private Collection<VocabWord> words;
	private PriorityQueue<VocabWord> heap = new PriorityQueue<>();

	public void build() {
		heap.addAll(words);

		int size = words.size();
		//build huffman tree
		for(int i = 0 ; i< words.size() - 1; i++) {
			VocabWord word1 = heap.poll();
			VocabWord word2 = heap.poll();
			if(word1 == null || word2 == null)
				break;
			double count = word1.getWordFrequency() + word2.getWordFrequency();
			VocabWord newWord = new VocabWord(count,VocabWord.PARENT_NODE);

			word1.setCode(0);
			word2.setCode(1);

			word1.setParent(newWord);
			word2.setParent(newWord);

			newWord.setLeft(word1);
			newWord.setRight(word2);
			int idx = i + size;
			newWord.setIndex(idx);
			heap.add(newWord);
		}

	
	while(!heap.isEmpty()) {
		heap.poll().createLinks();
	}




	}




	private int[] plus (int[] addTo,int add) {
		int[] copy = new int[addTo.length + 1];
		for(int c = 0; c < addTo.length; c++)
			copy[c] = addTo[c];
		copy[addTo.length] = add;
		return copy;
	}


}
