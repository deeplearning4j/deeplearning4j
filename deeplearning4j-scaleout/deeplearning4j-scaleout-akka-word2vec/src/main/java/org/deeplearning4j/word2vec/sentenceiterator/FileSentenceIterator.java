package org.deeplearning4j.word2vec.sentenceiterator;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.Queue;
import java.util.zip.GZIPInputStream;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;

@SuppressWarnings("unchecked")
public class FileSentenceIterator extends BaseSentenceIterator {

	/*
	 * Used as a pair for when
	 * the number of sentences is not known
	 */
	private Iterator<File> fileIterator;
	private Queue<String> cache;
	private LineIterator currLineIterator;
	private File dir;

	public FileSentenceIterator(SentencePreProcessor preProcessor,File dir) {
		super(preProcessor);
		this.dir = dir;
		cache = new java.util.concurrent.ConcurrentLinkedDeque<>();
		fileIterator = FileUtils.iterateFiles(dir, null, true);

	}

	public FileSentenceIterator(File dir) {
		this(null,dir);
	}


	@Override
	public synchronized  String nextSentence() {
		String ret = null;
		if(!cache.isEmpty()) {
			ret = cache.poll();
			if(preProcessor != null)
				ret = preProcessor.preProcess(ret);
			return ret;
		}
		else {

			if(currLineIterator == null || !currLineIterator.hasNext())
				nextLineIter();

			for(int i = 0; i < 100000; i++) {
				if(currLineIterator != null && currLineIterator.hasNext()) 
					cache.add(currLineIterator.nextLine());

				else
					break;
			}

			if(!cache.isEmpty()) {
				ret = cache.poll();
				if(preProcessor != null)
					ret = preProcessor.preProcess(ret);
				return ret;
			}

		}


		if(ret == null) {
			if(!cache.isEmpty())
				ret = cache.poll();
			else
				return null;
		}
		return ret;

	}



	private synchronized void nextLineIter() {
		if(fileIterator.hasNext()) {
			try {
				File next = fileIterator.next();
				if(next.getAbsolutePath().endsWith(".gz")) {
					if(currLineIterator != null)
						currLineIterator.close();
					currLineIterator = IOUtils.lineIterator(new BufferedInputStream(new GZIPInputStream(new FileInputStream(next))), "UTF-8");

				}
				else {
					if(currLineIterator != null)
						currLineIterator.close();
					currLineIterator = FileUtils.lineIterator(next);

				}
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
	}

	@Override
	public synchronized boolean hasNext() {
		return currLineIterator != null && currLineIterator.hasNext() || fileIterator.hasNext() || !cache.isEmpty();
	}


	@Override
	public void reset() {
		fileIterator = FileUtils.iterateFiles(dir, null, true);


	}



}
