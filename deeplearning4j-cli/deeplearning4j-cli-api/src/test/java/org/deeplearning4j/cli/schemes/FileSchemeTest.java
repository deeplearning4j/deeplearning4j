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

package org.deeplearning4j.cli.schemes;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.writable.Writable;
import org.deeplearning4j.cli.api.schemes.Scheme;
import org.deeplearning4j.cli.api.schemes.Schemes;
import org.deeplearning4j.cli.api.schemes.test.BaseSchemeTest;
import org.deeplearning4j.cli.files.FileScheme;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author sonali
 */
public class FileSchemeTest extends BaseSchemeTest {
    Scheme fileScheme = Schemes.getScheme("file");

    @Override
    public Scheme getScheme() {
        return fileScheme;
    }

    @Override
    public void initScheme() {

    }

}
