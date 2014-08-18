package org.deeplearning4j.word2vec.similarity;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.util.SetUtils;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.util.Util;


/**
 * Semantic similarity score 
 * on whether 2 articles are the same.
 * The cut off number is going to be low.
 * Experimentation is usually going to be required
 * to find the right cut off point for whether an article is "similar" or not
 * The more data you train on for word2vec the more accurate the classifier is.
 * for whether 2 articles are similar.
 * 
 * This is based on a few factors: one is 
 * the neural word embeddings from word2vec
 * 
 * The other is the percent of words that intersect in the article.
 * 
 * This comes out with a similarity score measured by the distance in the
 * word2vec vector space are from each other alongside content overlapping in
 * the articles.
 * 
 * @author Adam Gibson
 *
 */
public class Word2VecSimilarity {


	private Word2Vec vec;
	private String words1;
	private String words2;
	private float distance;
	
	public Word2VecSimilarity(String words1,String words2,Word2Vec vec) {
		this.words1 = words1;
		this.words2 = words2;
		this.vec = vec;
	}

	public void calc() {

		
		//calculate word frequencies
		WordMetaData d1 = new WordMetaData(vec,words1);
		WordMetaData d2 = new WordMetaData(vec,words2);
		d1.calc();
		d2.calc();

		//all the words occurring in each article
		Set<String> vocab = SetUtils.union(d1.getWordCounts().keySet(), d2.getWordCounts().keySet());
		
		//remove stop words
		Set<String> remove = new HashSet<String>();
		
		for(String word : vocab) {
			if(Util.matchesAnyStopWord(vec.getStopWords(),word))
				remove.add(word);
		}
		vocab.removeAll(remove);
		Set<String> inter = SetUtils.intersection(d1.getWordCounts().keySet(), d2.getWordCounts().keySet());
		inter.removeAll(remove);
		
		//words to be processed: need indexing
		List<String> wordList = new ArrayList<>(vocab);

		//the word embeddings (each row is a word)
		INDArray a1Matrix = NDArrays.create(wordList.size(), vec.getLayerSize());
        INDArray a2Matrix = NDArrays.create(wordList.size(), vec.getLayerSize());

		for(int i = 0; i < wordList.size(); i++) {
			if(d1.getWordCounts().getCount(wordList.get(i)) > 0) {
				a1Matrix.putRow(i,vec.getWordVectorMatrix(wordList.get(i)));
			}
			else 
				a1Matrix.putRow(i, NDArrays.zeros(vec.getLayerSize()));

			if(d2.getWordCounts().getCount(wordList.get(i)) > 0) {
				a2Matrix.putRow(i,vec.getWordVectorMatrix(wordList.get(i)));

			}
			else 
				a2Matrix.putRow(i, NDArrays.zeros(vec.getLayerSize()));



		}
		
		//percent of words that overlap
		float wordSim = (float) inter.size() / (float) wordList.size();
		//cosine similarity of the word embeddings * percent of words that overlap (this is a weight to add a decision boundary)
        float finalScore  = (float) Transforms.cosineSim(a1Matrix, a2Matrix) * wordSim;
		//threshold is >= 0.05 for any articles that are similar
		distance = finalScore;
	} 





	public double getDistance() {
		return distance;
	}

	

	

}
