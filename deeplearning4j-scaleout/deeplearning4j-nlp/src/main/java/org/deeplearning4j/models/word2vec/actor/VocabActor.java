package org.deeplearning4j.models.word2vec.actor;

import java.io.InputStream;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.compress.utils.IOUtils;
import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.models.word2vec.StreamWork;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.VocabWork;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
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
	private transient TokenizerFactory tokenizer;
	private List<String> stopWords;
	private AtomicLong lastUpdate;
	private VocabCache cache;
	private int minWordFrequency;




	public VocabActor(TokenizerFactory tokenizer,  VocabCache cache, int layerSize,List<String> stopWords,AtomicLong lastUpdate,int minWordFrequency) {
		super();
		this.tokenizer = tokenizer;
		this.stopWords = stopWords;
		this.lastUpdate = lastUpdate;
		this.cache = cache;
		this.minWordFrequency = minWordFrequency;
	}




	@Override
	public void onReceive(Object message) throws Exception {
		if(message  instanceof VocabWork) {
			VocabWork work = (VocabWork) message;
			if(work.getWork() == null || work.getWork().isEmpty())
				return;
			work.getCount().incrementAndGet();
			String sentence = work.getWork();

			if(sentence.isEmpty()) {
				work.countDown();
				return;
			}
			Tokenizer t = tokenizer.create(sentence);

			while(t.hasMoreTokens())  {
				String token = t.nextToken();
				if(stopWords.contains(token))
					token = "STOP";
				cache.incrementWordCount(token);
				//note that for purposes of word frequency, the
				//internal vocab and the final vocab
				//at the class level contain the same references
				if(!Util.matchesAnyStopWord(stopWords,token) && token != null && !token.isEmpty()) {
					if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
						VocabWord word = new VocabWord(cache.wordFrequency(token),token);
						int idx = cache.numWords();
						word.setIndex(idx);
						cache.putVocabWord(token,word);
					}


				}
				
				else  if(Util.matchesAnyStopWord(stopWords,token) && token != null && !token.isEmpty()) {
					token = "STOP";
					if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
						VocabWord word = new VocabWord(cache.wordFrequency(token),token);
						int idx = cache.numWords();
						word.setIndex(idx);
						cache.putVocabWord(token,word);
					}


				}


			}


			work.countDown();

			lastUpdate.getAndSet(System.currentTimeMillis());

		}


		else if(message instanceof StreamWork) {
			StreamWork work = (StreamWork) message;
			InputStream is = work.getIs();

			Tokenizer t = tokenizer.create(is);

			while(t.hasMoreTokens())  {
				String token = t.nextToken();
				if(stopWords.contains(token))
					token = "STOP";
				cache.incrementWordCount(token);
				//note that for purposes of word frequency, the
				//internal vocab and the final vocab
				//at the class level contain the same references
				if(!Util.matchesAnyStopWord(stopWords,token) && token != null && !token.isEmpty()) {
					if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
						VocabWord word = new VocabWord(cache.wordFrequency(token),token);
						int idx = cache.numWords();
						word.setIndex(idx);
						cache.putVocabWord(token,word);
					}
					
					else  if(Util.matchesAnyStopWord(stopWords,token) && token != null && !token.isEmpty()) {
						token = "STOP";
						if(!cache.containsWord(token) && cache.wordFrequency(token) >= minWordFrequency) {
							VocabWord word = new VocabWord(cache.wordFrequency(token),token);
							int idx = cache.numWords();
							word.setIndex(idx);
							cache.putVocabWord(token,word);
						}


					}

				}


			}

			IOUtils.closeQuietly(is);
			work.countDown();

			lastUpdate.getAndSet(System.currentTimeMillis());


		}



		else
			unhandled(message);
	}





}
