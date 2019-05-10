/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
