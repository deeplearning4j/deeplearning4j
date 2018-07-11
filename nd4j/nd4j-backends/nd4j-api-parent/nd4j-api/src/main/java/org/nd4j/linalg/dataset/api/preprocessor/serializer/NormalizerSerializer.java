package org.nd4j.linalg.dataset.api.preprocessor.serializer;

import lombok.NonNull;
import lombok.Value;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility for serializing and unserializing {@link Normalizer} instances.
 *
 * @author Ede Meijer
 */
public class NormalizerSerializer {

    private static final String HEADER = "NORMALIZER";
    private static NormalizerSerializer defaultSerializer;

    private List<NormalizerSerializerStrategy> strategies = new ArrayList<>();

    /**
     * Serialize a normalizer to the given file
     *
     * @param normalizer the normalizer
     * @param file       the destination file
     * @throws IOException
     */
    public void write(@NonNull Normalizer normalizer, @NonNull File file) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(file))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a normalizer to the given file path
     *
     * @param normalizer the normalizer
     * @param path       the destination file path
     * @throws IOException
     */
    public void write(@NonNull Normalizer normalizer, @NonNull String path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(new FileOutputStream(path))) {
            write(normalizer, out);
        }
    }

    /**
     * Serialize a normalizer to an output stream
     *
     * @param normalizer the normalizer
     * @param stream     the output stream to write to
     * @throws IOException
     */
    public void write(@NonNull Normalizer normalizer, @NonNull OutputStream stream) throws IOException {
        NormalizerSerializerStrategy strategy = getStrategy(normalizer);

        writeHeader(stream, Header.fromStrategy(strategy));
        //noinspection unchecked
        strategy.write(normalizer, stream);
    }

    /**
     * Restore a normalizer from the given path
     *
     * @param path path of the file containing a serialized normalizer
     * @return the restored normalizer
     * @throws IOException
     */
    public <T extends Normalizer> T restore(@NonNull String path) throws Exception {
        try (InputStream in = new BufferedInputStream(new FileInputStream(path))) {
            return restore(in);
        }
    }

    /**
     * Restore a normalizer from the given file
     *
     * @param file the file containing a serialized normalizer
     * @return the restored normalizer
     * @throws IOException
     */
    public <T extends Normalizer> T restore(@NonNull File file) throws Exception {
        try (InputStream in = new BufferedInputStream(new FileInputStream(file))) {
            return restore(in);
        }
    }

    /**
     * Restore a normalizer from an input stream
     *
     * @param stream a stream of serialized normalizer data
     * @return the restored normalizer
     * @throws IOException
     */
    public <T extends Normalizer> T restore(@NonNull InputStream stream) throws Exception {
        Header header = parseHeader(stream);

        //noinspection unchecked
        return (T) getStrategy(header).restore(stream);
    }

    /**
     * Get the default serializer configured with strategies for the built-in normalizer implementations
     *
     * @return the default serializer
     */
    public static NormalizerSerializer getDefault() {
        if (defaultSerializer == null) {
            defaultSerializer = new NormalizerSerializer().addStrategy(new StandardizeSerializerStrategy())
                    .addStrategy(new MinMaxSerializerStrategy())
                    .addStrategy(new MultiStandardizeSerializerStrategy())
                    .addStrategy(new MultiMinMaxSerializerStrategy())
                    .addStrategy(new ImagePreProcessingSerializerStrategy())
                    .addStrategy(new MultiHybridSerializerStrategy());
        }
        return defaultSerializer;
    }

    /**
     * Add a normalizer serializer strategy
     *
     * @param strategy the new strategy
     * @return self
     */
    public NormalizerSerializer addStrategy(@NonNull NormalizerSerializerStrategy strategy) {
        strategies.add(strategy);
        return this;
    }

    /**
     * Get a serializer strategy the given normalizer
     *
     * @param normalizer the normalizer to find a compatible serializer strategy for
     * @return the compatible strategy
     */
    private NormalizerSerializerStrategy getStrategy(Normalizer normalizer) {
        for (NormalizerSerializerStrategy strategy : strategies) {
            if (strategySupportsNormalizer(strategy, normalizer.getType(), normalizer.getClass())) {
                return strategy;
            }
        }
        throw new RuntimeException(String.format(
                "No serializer strategy found for normalizer of class %s. If this is a custom normalizer, you probably "
                        + "forgot to register a corresponding custom serializer strategy with this serializer.",
                normalizer.getClass()));
    }

    /**
     * Get a serializer strategy the given serialized file header
     *
     * @param header the header to find the associated serializer strategy for
     * @return the compatible strategy
     */
    private NormalizerSerializerStrategy getStrategy(Header header) throws Exception {
        if (header.normalizerType.equals(NormalizerType.CUSTOM)) {
            return header.customStrategyClass.newInstance();
        }
        for (NormalizerSerializerStrategy strategy : strategies) {
            if (strategySupportsNormalizer(strategy, header.normalizerType, null)) {
                return strategy;
            }
        }
        throw new RuntimeException("No serializer strategy found for given header " + header);
    }

    /**
     * Check if a serializer strategy supports a normalizer. If the normalizer is a custom opType, it checks if the
     * supported normalizer class matches.
     *
     * @param strategy
     * @param normalizerType
     * @param normalizerClass
     * @return whether the strategy supports the normalizer
     */
    private boolean strategySupportsNormalizer(NormalizerSerializerStrategy strategy, NormalizerType normalizerType,
                                               Class<? extends Normalizer> normalizerClass) {
        if (!strategy.getSupportedType().equals(normalizerType)) {
            return false;
        }
        if (strategy.getSupportedType().equals(NormalizerType.CUSTOM)) {
            // Strategy should be instance of CustomSerializerStrategy
            if (!(strategy instanceof CustomSerializerStrategy)) {
                throw new IllegalArgumentException(
                        "Strategies supporting CUSTOM opType must be instance of CustomSerializerStrategy, got"
                                + strategy.getClass());
            }
            return ((CustomSerializerStrategy) strategy).getSupportedClass().equals(normalizerClass);
        }
        return true;
    }

    /**
     * Parse the data header
     *
     * @param stream the input stream
     * @return the parsed header with information about the contents
     * @throws IOException
     * @throws IllegalArgumentException if the data format is invalid
     */
    private Header parseHeader(InputStream stream) throws IOException, ClassNotFoundException {
        DataInputStream dis = new DataInputStream(stream);
        // Check if the stream starts with the expected header
        String header = dis.readUTF();
        if (!header.equals(HEADER)) {
            throw new IllegalArgumentException(
                    "Could not restore normalizer: invalid header. If this normalizer was saved with a opType-specific "
                            + "strategy like StandardizeSerializerStrategy, use that class to restore it as well.");
        }

        // The next byte is an integer indicating the version
        int version = dis.readInt();

        // Right now, we only support version 1
        if (version != 1) {
            throw new IllegalArgumentException("Could not restore normalizer: invalid version (" + version + ")");
        }
        // The next value is a string indicating the normalizer opType
        NormalizerType type = NormalizerType.valueOf(dis.readUTF());
        if (type.equals(NormalizerType.CUSTOM)) {
            // For custom serializers, the next value is a string with the class opName
            String strategyClassName = dis.readUTF();
            //noinspection unchecked
            return new Header(type, (Class<? extends NormalizerSerializerStrategy>) Class.forName(strategyClassName));
        } else {
            return new Header(type, null);
        }
    }

    /**
     * Write the data header
     *
     * @param stream the output stream
     * @param header the header to write
     * @throws IOException
     */
    private void writeHeader(OutputStream stream, Header header) throws IOException {
        DataOutputStream dos = new DataOutputStream(stream);
        dos.writeUTF(HEADER);

        // Write the current version
        dos.writeInt(1);

        // Write the normalizer opType
        dos.writeUTF(header.normalizerType.toString());

        // If the header contains a custom class opName, write that too
        if (header.customStrategyClass != null) {
            dos.writeUTF(header.customStrategyClass.getName());
        }
    }

    /**
     * Represents the header of a serialized normalizer file
     */
    @Value
    private static class Header {
        NormalizerType normalizerType;
        Class<? extends NormalizerSerializerStrategy> customStrategyClass;

        public static Header fromStrategy(NormalizerSerializerStrategy strategy) {
            if (strategy instanceof CustomSerializerStrategy) {
                return new Header(strategy.getSupportedType(), strategy.getClass());
            } else {
                return new Header(strategy.getSupportedType(), null);
            }
        }
    }
}
