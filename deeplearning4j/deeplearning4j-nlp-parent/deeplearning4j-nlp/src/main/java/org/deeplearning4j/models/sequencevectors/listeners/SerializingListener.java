package org.deeplearning4j.models.sequencevectors.listeners;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.Semaphore;

/**
 *
 * This is example VectorsListener implementation. It can be used to serialize models in the middle of training process
 *
 * @author raver119@gmail.com
 */
public class SerializingListener<T extends SequenceElement> implements VectorsListener<T> {
    private File targetFolder = new File("./");
    private String modelPrefix = "Model_";
    private boolean useBinarySerialization = true;
    private ListenerEvent targetEvent = ListenerEvent.EPOCH;
    private int targetFrequency = 100000;

    private Semaphore locker = new Semaphore(1);

    protected SerializingListener() {}

    /**
     * This method is called prior each processEvent call, to check if this specific VectorsListener implementation is viable for specific event
     *
     * @param event
     * @param argument
     * @return TRUE, if this event can and should be processed with this listener, FALSE otherwise
     */
    @Override
    public boolean validateEvent(ListenerEvent event, long argument) {
        try {
            /**
             * please note, since sequence vectors are multithreaded we need to stop processed while model is being saved
             */
            locker.acquire();

            if (event == targetEvent && argument % targetFrequency == 0) {
                return true;
            } else
                return false;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            locker.release();
        }
    }

    /**
     * This method is called at each epoch end
     *
     * @param event
     * @param sequenceVectors
     * @param argument
     */
    @Override
    public void processEvent(ListenerEvent event, SequenceVectors<T> sequenceVectors, long argument) {
        try {
            locker.acquire();

            SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");

            StringBuilder builder = new StringBuilder(targetFolder.getAbsolutePath());
            builder.append("/").append(modelPrefix).append("_").append(sdf.format(new Date())).append(".seqvec");
            File targetFile = new File(builder.toString());

            if (useBinarySerialization) {
                SerializationUtils.saveObject(sequenceVectors, targetFile);
            } else {
                throw new UnsupportedOperationException("Not implemented yet");
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            locker.release();
        }
    }

    public static class Builder<T extends SequenceElement> {
        private File targetFolder = new File("./");
        private String modelPrefix = "Model_";
        private boolean useBinarySerialization = true;
        private ListenerEvent targetEvent = ListenerEvent.EPOCH;
        private int targetFrequency = 100000;

        public Builder(ListenerEvent targetEvent, int frequency) {
            this.targetEvent = targetEvent;
            this.targetFrequency = frequency;
        }

        /**
         * This method allows you to define template for file names that will be created during serialization
         * @param reallyUse
         * @return
         */
        public Builder<T> setFilenamePrefix(boolean reallyUse) {
            this.useBinarySerialization = reallyUse;
            return this;
        }

        /**
         * This method specifies target folder where models should be saved
         *
         * @param folder
         * @return
         */
        public Builder<T> setTargetFolder(@NonNull String folder) {
            this.setTargetFolder(new File(folder));
            return this;
        }

        /**
         * This method specifies target folder where models should be saved
         *
         * @param folder
         * @return
         */
        public Builder<T> setTargetFolder(@NonNull File folder) {
            if (!folder.exists() || !folder.isDirectory())
                throw new IllegalStateException("Target folder must exist!");
            this.targetFolder = folder;
            return this;
        }

        /**
         * This method returns new SerializingListener instance
         *
         * @return
         */
        public SerializingListener<T> build() {
            SerializingListener<T> listener = new SerializingListener<>();
            listener.modelPrefix = this.modelPrefix;
            listener.targetFolder = this.targetFolder;
            listener.useBinarySerialization = this.useBinarySerialization;
            listener.targetEvent = this.targetEvent;
            listener.targetFrequency = this.targetFrequency;

            return listener;
        }
    }
}
