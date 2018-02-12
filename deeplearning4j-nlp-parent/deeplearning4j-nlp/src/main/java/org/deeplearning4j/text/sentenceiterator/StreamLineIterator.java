package org.deeplearning4j.text.sentenceiterator;

import lombok.NonNull;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Simple class suitable for iterating over InputStreams as over lines of strings
 *
 * Please note, this class is NOT thread safe
 *
 * @author raver119@gmail.com
 */
public class StreamLineIterator implements SentenceIterator {
    private DocumentIterator iterator;
    private int linesToFetch;
    private final LinkedBlockingQueue<String> buffer = new LinkedBlockingQueue<>();
    private SentencePreProcessor preProcessor;

    private BufferedReader currentReader;

    protected Logger logger = LoggerFactory.getLogger(StreamLineIterator.class);

    private StreamLineIterator(DocumentIterator iterator) {
        this.iterator = iterator;
    }

    private void fetchLines(int linesToFetch) {
        String line = "";
        int cnt = 0;
        try {
            while (cnt < linesToFetch && (line = currentReader.readLine()) != null) {
                buffer.add(line);
                cnt++;
            }

            // in this case we nullify currentReader as sign of finished reading
            if (line == null) {
                currentReader.close();
                currentReader = null;
            }
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    @Override
    public String nextSentence() {
        if (buffer.size() < linesToFetch) {
            // prefetch
            if (currentReader != null) {
                fetchLines(linesToFetch);
            } else if (this.iterator.hasNext()) {
                currentReader = new BufferedReader(new InputStreamReader(iterator.nextDocument()));
                fetchLines(linesToFetch);
            }
        }

        // actually its the same. You get string or you get null as result of poll, if buffer is empty after prefetch try
        if (buffer.isEmpty())
            return null;
        else
            return buffer.poll();
    }

    @Override
    public boolean hasNext() {
        try {
            return !buffer.isEmpty() || iterator.hasNext() || (currentReader != null && currentReader.ready());
        } catch (IOException e) {
            // this exception is possible only at currentReader.ready(), so it means that it's definitely NOT ready
            return false;
        }
    }

    @Override
    public void reset() {
        iterator.reset();
    }

    @Override
    public void finish() {
        buffer.clear();
    }

    @Override
    public SentencePreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    @Override
    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    public static class Builder {
        private DocumentIterator iterator;
        private int linesToFetch = 50;
        private SentencePreProcessor preProcessor;

        public Builder(@NonNull final InputStream stream) {
            this(new DocumentIterator() {
                private final InputStream onlyStream = stream;
                private AtomicBoolean isConsumed = new AtomicBoolean(false);

                @Override
                public boolean hasNext() {
                    return !isConsumed.get();
                }

                @Override
                public InputStream nextDocument() {
                    isConsumed.set(true);
                    return this.onlyStream;
                }

                @Override
                public void reset() {
                    isConsumed.set(false);
                    try {
                        this.onlyStream.reset();
                    } catch (IOException e) {
                        e.printStackTrace();
                        throw new RuntimeException(e);
                    }
                }
            });
        }

        public Builder(@NonNull DocumentIterator iterator) {
            this.iterator = iterator;
        }

        public Builder setFetchSize(int linesToFetch) {
            this.linesToFetch = linesToFetch;
            return this;
        }

        public Builder setPreProcessor(SentencePreProcessor preProcessor) {
            this.preProcessor = preProcessor;
            return this;
        }

        public StreamLineIterator build() {
            StreamLineIterator lineIterator = new StreamLineIterator(this.iterator);
            lineIterator.linesToFetch = linesToFetch;
            lineIterator.setPreProcessor(preProcessor);

            return lineIterator;
        }
    }
}
