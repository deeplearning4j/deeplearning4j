package org.deeplearning4j.models.word2vec;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.Stack;

import org.deeplearning4j.berkeley.Triple;
import org.nd4j.linalg.util.ArrayUtil;
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
	private PriorityQueue<VocabWord> heap;

	public void build() {
		heap = new PriorityQueue<>(words.size());
		heap.addAll(words);

		List<VocabWord> wordRefs = new ArrayList<>();
		wordRefs.addAll(words);

		int size = words.size();
		//build huffman tree

		for(int i = 0; i < words.size() - 1; i++) {
			VocabWord word1 = heap.poll();
			VocabWord word2 = heap.poll();

			double count = word1.getWordFrequency() + word2.getWordFrequency();
			VocabWord newWord = new VocabWord(count,VocabWord.PARENT_NODE);

			word1.setCode(0);
			word2.setCode(1);

			word1.setParent(newWord);
			word2.setParent(newWord);

			newWord.setLeft(word1);
			newWord.setRight(word2);

			newWord.setIndex(i  + words.size());

			heap.add(newWord);

			wordRefs.add(newWord);



		}

		Stack<Triple<VocabWord,int[],int[]>> stack = new Stack<>();
		stack.add(new Triple<>(heap.poll(),new int[0],new int[0]));
		while(!stack.isEmpty()) {
			Triple<VocabWord,int[],int[]> triple = stack.pop();
			VocabWord node = triple.getFirst();
			int[] codes = triple.getSecond();
			int[] points = triple.getThird();
			if(node.getIndex() < words.size()) {
				node.setCodes(ArrayUtil.copy(codes));
				node.setPoints(ArrayUtil.copy(points));
			}
			else {
				points = plus(points,node.getIndex() - words.size());
				stack.push(new Triple<>(node.getLeft(),plus(codes,0),ArrayUtil.copy(points)));
				stack.push(new Triple<>(node.getRight(),plus(codes,1),ArrayUtil.copy(points)));
			}

		}



	}




	private int[] plus (int[] addTo,int add) {
	 if(addTo == null || addTo.length < 1)
		 return new int[] {add};
		int[] copy = new int[addTo.length + 1];
		for(int c = 0; c < addTo.length; c++)
			copy[c] = addTo[c];
		copy[addTo.length] = add;
		return copy;
	}


}
