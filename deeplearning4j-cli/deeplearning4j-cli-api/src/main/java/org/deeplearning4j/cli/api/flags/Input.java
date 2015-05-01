/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.cli.api.flags;

import java.net.URI;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;

/**
 * Input flag for loading input data for the model
 *
 * @author sonali
 */
public abstract class Input extends BaseIOFlag {

    @Override
    public <E> E value(String value) throws Exception {
        URI uri = URI.create(value);
        String path = uri.getPath();
        String extension = path.substring(path.lastIndexOf(".") + 1);

        return (E) createReader(uri);
    }

    @Override
    protected RecordWriter createWriter(URI uri) {
        return null;
    }

    @Override
    protected RecordReader createReader(URI uri) {
        return null;
    }
}
