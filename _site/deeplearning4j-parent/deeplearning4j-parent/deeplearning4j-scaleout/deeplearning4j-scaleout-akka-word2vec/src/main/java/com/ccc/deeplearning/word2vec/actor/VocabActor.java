package com.ccc.deeplearning.word2vec.actor;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.UntypedActor;
import akka.japi.Creator;

import com.ccc.deeplearning.berkeley.Counter;
import com.ccc.deeplearning.word2vec.VocabWord;
import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.util.Util;
import com.ccc.deeplearning.word2vec.viterbi.Index;

public class VocabActor extends UntypedActor {

	private static Logger log = LoggerFactory.getLogger(VocabActor.class);

	@Override
	public void onReceive(Object message) throws Exception {
		if(message instanceof VocabMessage) {
			VocabMessage m = (VocabMessage) message;
			Counter<String> rawVocab = m.getRawVocab();
			List<String> tokens = m.getTokens();
			List<String> stopWords = m.getStopWords();
			int minWordFrequency = m.getMinWordFrequency();
			Index wordIndex = m.getWordIndex();
			Map<String,VocabWord> vocab = m.getVocab();
			int layerSize = m.getLayerSize();
			AtomicLong lastUpdate = m.getChangeTracker();
			
			for(String token : tokens) {
				if(stopWords.contains(token))
					token = "STOP";
				rawVocab.incrementCount(token,1.0);
				//note that for purposes of word frequency, the 
				//internal vocab and the final vocab
				//at the class level contain the same references
				if(rawVocab.getCount(token) >= minWordFrequency && !Util.matchesAnyStopWord(stopWords,token)) {
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
