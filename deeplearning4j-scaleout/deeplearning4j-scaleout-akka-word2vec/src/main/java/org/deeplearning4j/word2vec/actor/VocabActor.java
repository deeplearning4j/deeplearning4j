package org.deeplearning4j.word2vec.actor;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.word2vec.VocabWord;
import org.deeplearning4j.word2vec.tokenizer.Tokenizer;
import org.deeplearning4j.word2vec.tokenizer.TokenizerFactory;
import org.deeplearning4j.word2vec.util.Util;
import org.deeplearning4j.util.Index;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.UntypedActor;


public class VocabActor extends UntypedActor {

	private static Logger log = LoggerFactory.getLogger(VocabActor.class);
	private TokenizerFactory tokenizer;
	private Index wordIndex;
	private int minWordFrequency;
	private Map<String,VocabWord> vocab;
	private int layerSize;
	private List<String> stopWords;
	private Counter<String> rawVocab;
	private AtomicLong lastUpdate;




	public VocabActor(TokenizerFactory tokenizer, Index wordIndex,
			int minWordFrequency, Map<String, VocabWord> vocab, int layerSize,
			List<String> stopWords, Counter<String> rawVocab,AtomicLong lastUpdate) {
		super();
		this.tokenizer = tokenizer;
		this.wordIndex = wordIndex;
		this.minWordFrequency = minWordFrequency;
		this.vocab = vocab;
		this.layerSize = layerSize;
		this.stopWords = stopWords;
		this.rawVocab = rawVocab;
		this.lastUpdate = lastUpdate;
	}




	@Override
	public void onReceive(Object message) throws Exception {
		if(message  instanceof String) {
			String sentence = message.toString();
			Tokenizer t = tokenizer.create(sentence);
			List<String> tokens = new ArrayList<String>();
			while(t.hasMoreTokens())
				tokens.add(t.nextToken());
			getSelf().tell(tokens,getSelf());

		}

		else if(message instanceof Collection) {
			Collection<String> tokens = (Collection<String>) message;
			for(String token : tokens) {
				if(stopWords.contains(token))
					token = "STOP";
				rawVocab.incrementCount(token,1.0);
				//note that for purposes of word frequency, the 
				//internal vocab and the final vocab
				//at the class level contain the same references
				if(!Util.matchesAnyStopWord(stopWords,token)) {
					if(!vocab.containsKey(token)) {
						VocabWord word = new VocabWord(rawVocab.getCount(token),layerSize);
						word.setIndex(vocab.size());
						vocab.put(token, word);
						wordIndex.add(token);

					}


				}


			}

			lastUpdate.getAndSet(System.currentTimeMillis());

		}

		else 
			unhandled(message);
	}





}
