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

package org.deeplearning4j.text.corpora.sentiwordnet;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Vector;

import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.util.JCasUtil;
import org.canova.api.util.ClassPathResource;
import org.cleartk.token.type.Sentence;
import org.cleartk.token.type.Token;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;

import com.google.common.collect.Sets;
/**
 * Based on SentiWordnet
 * @author Adam Gibson
 *
 */
public class SWN3 implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -2614454572930777658L;
	private HashMap<String, Double> _dict;
	private Set<String> negationWords = Sets.newHashSet("could","would","should","not","isn't","aren't","wasn't","weren't","haven't","doesn't","didn't","don't");
	private AnalysisEngine analysisEngine;
	
	public SWN3() throws Exception {
		this(UimaTokenizerFactory.defaultAnalysisEngine());
	}

	public SWN3(AnalysisEngine analysisEngine) {
		this("/sentiment/sentiwordnet.txt");
		this.analysisEngine = analysisEngine;
	}
	
	public SWN3(String sentiWordNetPath) {

		_dict = new HashMap<String, Double>();
		HashMap<String, Vector<Double>> _temp = new HashMap<String, Vector<Double>>();

		ClassPathResource resource = new ClassPathResource(sentiWordNetPath);

		try{
			BufferedReader csv =  new BufferedReader(new InputStreamReader(resource.getInputStream()));
			String line = "";           
			while((line = csv.readLine()) != null) {
				if(line.isEmpty())
					continue;
				String[] data = line.split("\t");

				if(data[2].isEmpty() || data[3].isEmpty())
					continue;
				Double score = Double.parseDouble(data[2])-Double.parseDouble(data[3]);
				String[] words = data[4].split(" ");
				for(String w : words) {
					if(w.isEmpty())
						continue;

					String[] w_n = w.split("#");
					w_n[0] += "#"+data[0];
					int index = Integer.parseInt(w_n[1])-1;
					if(_temp.containsKey(w_n[0])) {
						Vector<Double> v = _temp.get(w_n[0]);
						if(index>v.size())
							for(int i = v.size();i<index; i++)
								v.add(0.0);
						v.add(index, score);
						_temp.put(w_n[0], v);
					}
					else {
						Vector<Double> v = new Vector<Double>();
						for(int i = 0;i<index; i++)
							v.add(0.0);
						v.add(index, score);
						_temp.put(w_n[0], v);
					}
				}
			}
			
			
			Set<String> temp = _temp.keySet();
			for (Iterator<String> iterator = temp.iterator(); iterator.hasNext(); ) {
				String word = iterator.next();
				Vector<Double> v = _temp.get(word);
				double score = 0.0;
				double sum = 0.0;
				for(int i = 0; i < v.size(); i++)
					score += ((double)1/(double)(i+1))*v.get(i);
				for(int i = 1; i<=v.size(); i++)
					sum += (double)1/(double)i;
				score /= sum;
				_dict.put(word, score);
			}
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}        
	}

	
	/**
	 * Classifies the given text
	 * @param text the text to classify
	 * @return the classification for the text
	 * @throws Exception
	 */
	public String classify(String text) throws Exception {
		return this.classForScore(score(text));
	}
	/**
	 * Scores the text
	 * @param words the text to score
	 * @return the score (polarity) for the text
	 * @throws Exception
	 */
	public double score(String words) throws Exception {
	 	CAS cas = analysisEngine.newCAS();
	 	cas.setDocumentText(words);
	 	analysisEngine.process(cas);
	 	return score(cas);
	}
	

	public String classForScore(Double score) {
		String sent = "neutral"; 
		if(score>=0.75)
			sent = "strong_positive";
		else if(score > 0.25 && score<=0.5)
			sent = "positive";
		else if(score > 0 && score>=0.25)
			sent = "weak_positive";
		else if(score < 0 && score>=-0.25)
			sent = "weak_negative";
		else if(score < -0.25 && score>=-0.5)
			sent = "negative";
		else if(score<=-0.75)
			sent = "strong_negative";
		return sent;
	}


	public String classify(CAS cas) throws CASException {
		return classForScore(score(cas));
	}



	public double scoreTokens(List<Token> tokens) {
		double totalScore = 0.0;
		Set<String> negativeWords = new HashSet<String>();
		double scoreForSentence = 0.0;
		for(Token token : tokens) {
			scoreForSentence += extract(token.getCoveredText().toLowerCase());
			if(negationWords.contains(token.getCoveredText())) {
				negativeWords.add(token.getCoveredText());
			}
		}
		//flip for context
		if(!negativeWords.isEmpty()) {
			scoreForSentence *= -1.0;
		}

		totalScore +=scoreForSentence;
		return totalScore;
	}



	public double score(CAS cas) throws CASException {
		double totalScore = 0.0;
		for(Sentence sentence : JCasUtil.select(cas.getJCas(),Sentence.class)) {
			totalScore += scoreTokens(JCasUtil.selectCovered(Token.class,sentence));
		}

		return totalScore;
	}


	public String classify(Sentence sentence) {
		double totalScore = 0.0;
		for(Token token : JCasUtil.selectCovered(Token.class,sentence)) {
			totalScore += extract(token.getCoveredText().toLowerCase());
		}
		return classForScore(totalScore);
	}
	
	
	public double score(Sentence sentence) {
		double totalScore = 0.0;
		for(Token token : JCasUtil.selectCovered(Token.class, sentence)) {
			totalScore += extract(token.getCoveredText().toLowerCase());
		}
		return totalScore;
	}


	public Double extract(String word) {
		double total = 0.0;
		if(_dict.get(word + "#n") != null)
			total = _dict.get(word + "#n") + total;
		if(_dict.get(word + "#a") != null)
			total = _dict.get(word + "#a") + total;
		if(_dict.get(word + "#r") != null)
			total = _dict.get(word + "#r") + total;
		if(_dict.get(word + "#v") != null)
			total = _dict.get(word + "#v") + total;
		return total;
	}
	
	
	public static void main(String[] args) {
		SWN3 swn = new SWN3("/sentiment/sentiwordnet.txt");
		System.out.println(swn.classForScore(swn.extract("sad")));

	}
}