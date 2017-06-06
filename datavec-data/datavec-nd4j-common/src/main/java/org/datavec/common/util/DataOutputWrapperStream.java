/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.common.util;

import lombok.AllArgsConstructor;

import java.io.DataOutput;
import java.io.IOException;
import java.io.OutputStream;

/**
 * Created by Alex on 02/06/2017.
 */
@AllArgsConstructor
public class DataOutputWrapperStream extends OutputStream {

    private DataOutput underlying;

    @Override
    public void write(int b) throws IOException {
        underlying.write(b);
    }
}
