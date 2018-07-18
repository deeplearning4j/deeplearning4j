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

package org.ansj.dic;

import org.ansj.dic.impl.File2Stream;
import org.ansj.dic.impl.Jar2Stream;
import org.ansj.dic.impl.Jdbc2Stream;
import org.ansj.dic.impl.Url2Stream;
import org.ansj.exception.LibraryException;

import java.io.InputStream;

/**
 * 将路径转换为流，如果你需要实现自己的加载器请实现这个类，使用这个类可能需要自己依赖第三方包，比如jdbc连接和nutz
 * 
 * @author ansj
 *
 */
public abstract class PathToStream {

    public static InputStream stream(String path) {
        try {
            if (path.startsWith("file://")) {
                return new File2Stream().toStream(path);
            } else if (path.startsWith("jdbc://")) {
                return new Jdbc2Stream().toStream(path);
            } else if (path.startsWith("jar://")) {
                return new Jar2Stream().toStream(path);
            } else if (path.startsWith("class://")) {
                ((PathToStream) Class.forName(path.substring(8).split("\\|")[0]).newInstance()).toStream(path);
            } else if (path.startsWith("http://") || path.startsWith("https://")) {
                return new Url2Stream().toStream(path);
            } else {
                return new File2Stream().toStream(path);
            }
        } catch (Exception e) {
            throw new LibraryException(e);
        }
        throw new LibraryException("not find method type in path " + path);
    }

    public abstract InputStream toStream(String path);

}
