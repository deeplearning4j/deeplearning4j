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
import org.deeplearning4j.word2vec.wordstore.VocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.UntypedActor;

/**
 * Individual actor for updating the vocab cache
 *
 * @author Adam Gibson
 */
public class VocabActor extends UntypedActor {

	private static Logger log = LoggerFactory.getLogger(VocabActor.class);
	private TokenizerFactory tokenizer;
	private int layerSize;
	private List<String> stopWords;
	private AtomicLong lastUpdate;
    private VocabCache cache;




	public VocabActor(TokenizerFactory tokenizer,  VocabCache cache, int layerSize,List<String> stopWords,AtomicLong lastUpdate) {
		super();
		this.tokenizer = tokenizer;
		this.layerSize = layerSize;
		this.stopWords = stopWords;
		this.lastUpdate = lastUpdate;
        this.cache = cache;
	}




	@Override
	public void onReceive(Object message) throws Exception {
		if(message  instanceof String) {
			String sentence = message.toString();
			Tokenizer t = tokenizer.create(sentence);
			List<String> tokens = new ArrayList<>();
			while(t.hasMoreTokens())
				tokens.add(t.nextToken());
			getSelf().tell(tokens,getSelf());

		}

		else if(message instanceof Collection) {
			Collection<String> tokens = (Collection<String>) message;
			for(String token : tokens) {
				if(stopWords.contains
                        (token))
					token = "STOP";
                cache.incrementWordCount(token);
                log.info("Incremented token " + token);
				//note that for purposes of word frequency, the
				//internal vocab and the final vocab
				//at the class level contain the same references
				if(!Util.matchesAnyStopWord(stopWords,token)) {
					if(!cache.containsWord(token)) {
						VocabWord word = new VocabWord(cache.wordFrequency(token),layerSize);
						word.setIndex(cache.numWords());
						cache.putVocabWord(token,word);
                        cache.addWordToIndex(cache.numWords(),token);

					}


				}


			}

			lastUpdate.getAndSet(System.currentTimeMillis());

		}

		else 
			unhandled(message);
	}





}
