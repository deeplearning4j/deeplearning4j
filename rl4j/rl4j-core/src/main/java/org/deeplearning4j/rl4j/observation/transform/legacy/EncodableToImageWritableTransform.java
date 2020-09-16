/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.observation.transform.legacy;

import org.bytedeco.javacv.Frame;
import org.datavec.api.transform.Operation;
import org.datavec.image.data.ImageWritable;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.rl4j.space.Encodable;

public class EncodableToImageWritableTransform implements Operation<Encodable, ImageWritable> {

    final static NativeImageLoader nativeImageLoader = new NativeImageLoader();

    @Override
    public ImageWritable transform(Encodable encodable) {
        return new ImageWritable(nativeImageLoader.asFrame(encodable.getData(), Frame.DEPTH_UBYTE));
    }

}
