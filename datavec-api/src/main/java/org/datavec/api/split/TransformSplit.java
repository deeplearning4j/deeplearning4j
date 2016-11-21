package org.datavec.api.split;

import lombok.NonNull;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

/**
 * InputSplit implementation that maps the URIs of a given BaseInputSplit to new URIs. Useful when features and labels
 * are in different files sharing a common naming scheme, and the name of the output file can be determined given the
 * name of the input file.
 *
 * @author Ede Meijer
 */
public class TransformSplit extends BaseInputSplit {
    private final BaseInputSplit sourceSplit;
    private final URITransform transform;

    /**
     * Apply a given transformation to the raw URI objects
     *
     * @param sourceSplit the split with URIs to transform
     * @param transform transform operation that returns a new URI based on an input URI
     * @throws URISyntaxException thrown if the transformed URI is malformed
     */
    public TransformSplit(
        @NonNull BaseInputSplit sourceSplit,
        @NonNull URITransform transform
    ) throws URISyntaxException {
        this.sourceSplit = sourceSplit;
        this.transform = transform;
        initialize();
    }

    /**
     * Static factory method, replace the string version of the URI with a simple search-replace pair
     *
     * @param sourceSplit the split with URIs to transform
     * @param search the string to search
     * @param replace the string to replace with
     * @throws URISyntaxException thrown if the transformed URI is malformed
     */
    public static TransformSplit ofSearchReplace(
        @NonNull BaseInputSplit sourceSplit,
        @NonNull String search,
        @NonNull String replace
    ) throws URISyntaxException {
        return new TransformSplit(sourceSplit, new URITransform() {
            @Override
            public URI apply(URI uri) throws URISyntaxException {
                return new URI(uri.toString().replace(search, replace));
            }
        });
    }

    private void initialize() throws URISyntaxException {
        length = sourceSplit.length();
        locations = new URI[sourceSplit.locations().length];
        URI[] sourceLocations = sourceSplit.locations();
        for (int i = 0; i < sourceLocations.length; i++) {
            locations[i] = transform.apply(sourceLocations[i]);
        }
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    public interface URITransform {
        URI apply(URI uri) throws URISyntaxException;
    }
}
