package org.datavec.api.split;

import org.junit.Test;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Collection;

import static java.util.Arrays.asList;
import static org.junit.Assert.assertArrayEquals;

/**
 * @author Ede Meijer
 */
public class TransformSplitTest {
    @Test
    public void testTransform() throws URISyntaxException {
        Collection<URI> inputFiles = asList(new URI("file:///foo/bar/../0.csv"), new URI("file:///foo/1.csv"));

        InputSplit SUT = new TransformSplit(new CollectionInputSplit(inputFiles), new TransformSplit.URITransform() {
            @Override
            public URI apply(URI uri) throws URISyntaxException {
                return uri.normalize();
            }
        });

        assertArrayEquals(new URI[] {new URI("file:///foo/0.csv"), new URI("file:///foo/1.csv")}, SUT.locations());
    }

    @Test
    public void testSearchReplace() throws URISyntaxException {
        Collection<URI> inputFiles = asList(new URI("file:///foo/1-in.csv"), new URI("file:///foo/2-in.csv"));

        InputSplit SUT = TransformSplit.ofSearchReplace(new CollectionInputSplit(inputFiles), "-in.csv", "-out.csv");

        assertArrayEquals(new URI[] {new URI("file:///foo/1-out.csv"), new URI("file:///foo/2-out.csv")},
                        SUT.locations());
    }
}
