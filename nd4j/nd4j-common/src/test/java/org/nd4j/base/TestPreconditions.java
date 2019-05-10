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

package org.nd4j.base;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public class TestPreconditions {

    @Test
    public void testPreconditions(){

        Preconditions.checkArgument(true);
        try{
            Preconditions.checkArgument(false);
        } catch (IllegalArgumentException e){
            assertNull(e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here", 10);
        try{
            Preconditions.checkArgument(false, "Message %s here", 10);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there", 10, 20);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there", 10, 20);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there %s more", 10, 20, 30);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there %s more", 10, 20, 30);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here", 10L);
        try{
            Preconditions.checkArgument(false, "Message %s here", 10L);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there", 10L, 20L);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there", 10L, 20L);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there %s more", 10L, 20L, 30L);
        try{
            Preconditions.checkArgument(false, "Message %s here %s there %s more", 10L, 20L, 30L);
        } catch (IllegalArgumentException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkArgument(true, "Message %s here %s there %s more", "A", "B", "C");
        try{
            Preconditions.checkArgument(false, "Message %s here %s there %s more", "A", "B", "C");
        } catch (IllegalArgumentException e){
            assertEquals("Message A here B there C more", e.getMessage());
        }


    }

    @Test
    public void testPreconditionsMalformed(){

        //No %s:
        Preconditions.checkArgument(true, "This is malformed", "A", "B", "C");
        try{
            Preconditions.checkArgument(false, "This is malformed", "A", "B", "C");
        } catch (IllegalArgumentException e){
            assertEquals("This is malformed [A,B,C]", e.getMessage());
        }

        //More args than %s:
        Preconditions.checkArgument(true, "This is %s malformed", "A", "B", "C");
        try{
            Preconditions.checkArgument(false, "This is %s malformed", "A", "B", "C");
        } catch (IllegalArgumentException e){
            assertEquals("This is A malformed [B,C]", e.getMessage());
        }

        //No args
        Preconditions.checkArgument(true, "This is %s %s malformed");
        try{
            Preconditions.checkArgument(false, "This is %s %s malformed");
        } catch (IllegalArgumentException e){
            assertEquals("This is %s %s malformed", e.getMessage());
        }

        //More %s than args
        Preconditions.checkArgument(true, "This is %s %s malformed", "A");
        try{
            Preconditions.checkArgument(false, "This is %s %s malformed", "A");
        } catch (IllegalArgumentException e){
            assertEquals("This is A %s malformed", e.getMessage());
        }
    }


    @Test
    public void testPreconditionsState(){

        Preconditions.checkState(true);
        try{
            Preconditions.checkState(false);
        } catch (IllegalStateException e){
            assertNull(e.getMessage());
        }

        Preconditions.checkState(true, "Message %s here", 10);
        try{
            Preconditions.checkState(false, "Message %s here", 10);
        } catch (IllegalStateException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkState(true, "Message %s here %s there", 10, 20);
        try{
            Preconditions.checkState(false, "Message %s here %s there", 10, 20);
        } catch (IllegalStateException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkState(true, "Message %s here %s there %s more", 10, 20, 30);
        try{
            Preconditions.checkState(false, "Message %s here %s there %s more", 10, 20, 30);
        } catch (IllegalStateException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkState(true, "Message %s here", 10L);
        try{
            Preconditions.checkState(false, "Message %s here", 10L);
        } catch (IllegalStateException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkState(true, "Message %s here %s there", 10L, 20L);
        try{
            Preconditions.checkState(false, "Message %s here %s there", 10L, 20L);
        } catch (IllegalStateException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkState(true, "Message %s here %s there %s more", 10L, 20L, 30L);
        try{
            Preconditions.checkState(false, "Message %s here %s there %s more", 10L, 20L, 30L);
        } catch (IllegalStateException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkState(true, "Message %s here %s there %s more", "A", "B", "C");
        try{
            Preconditions.checkState(false, "Message %s here %s there %s more", "A", "B", "C");
        } catch (IllegalStateException e){
            assertEquals("Message A here B there C more", e.getMessage());
        }
    }

    @Test
    public void testPreconditionsMalformedState(){

        //No %s:
        Preconditions.checkState(true, "This is malformed", "A", "B", "C");
        try{
            Preconditions.checkState(false, "This is malformed", "A", "B", "C");
        } catch (IllegalStateException e){
            assertEquals("This is malformed [A,B,C]", e.getMessage());
        }

        //More args than %s:
        Preconditions.checkState(true, "This is %s malformed", "A", "B", "C");
        try{
            Preconditions.checkState(false, "This is %s malformed", "A", "B", "C");
        } catch (IllegalStateException e){
            assertEquals("This is A malformed [B,C]", e.getMessage());
        }

        //No args
        Preconditions.checkState(true, "This is %s %s malformed");
        try{
            Preconditions.checkState(false, "This is %s %s malformed");
        } catch (IllegalStateException e){
            assertEquals("This is %s %s malformed", e.getMessage());
        }

        //More %s than args
        Preconditions.checkState(true, "This is %s %s malformed", "A");
        try{
            Preconditions.checkState(false, "This is %s %s malformed", "A");
        } catch (IllegalStateException e){
            assertEquals("This is A %s malformed", e.getMessage());
        }
    }


    @Test
    public void testPreconditionsNull(){

        Preconditions.checkNotNull("");
        try{
            Preconditions.checkNotNull(null);
        } catch (NullPointerException e){
            assertNull(e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here", 10);
        try{
            Preconditions.checkNotNull(null, "Message %s here", 10);
        } catch (NullPointerException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here %s there", 10, 20);
        try{
            Preconditions.checkNotNull(null, "Message %s here %s there", 10, 20);
        } catch (NullPointerException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here %s there %s more", 10, 20, 30);
        try{
            Preconditions.checkNotNull(null, "Message %s here %s there %s more", 10, 20, 30);
        } catch (NullPointerException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here", 10L);
        try{
            Preconditions.checkNotNull(null, "Message %s here", 10L);
        } catch (NullPointerException e){
            assertEquals("Message 10 here", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here %s there", 10L, 20L);
        try{
            Preconditions.checkNotNull(null, "Message %s here %s there", 10L, 20L);
        } catch (NullPointerException e){
            assertEquals("Message 10 here 20 there", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here %s there %s more", 10L, 20L, 30L);
        try{
            Preconditions.checkNotNull(null, "Message %s here %s there %s more", 10L, 20L, 30L);
        } catch (NullPointerException e){
            assertEquals("Message 10 here 20 there 30 more", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here %s there %s more", "A", "B", "C");
        try{
            Preconditions.checkNotNull(null, "Message %s here %s there %s more", "A", "B", "C");
        } catch (NullPointerException e){
            assertEquals("Message A here B there C more", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here %s there %s more", new int[]{0,1}, new double[]{2.0, 3.0}, new boolean[]{true, false});
        try{
            Preconditions.checkNotNull(null, "Message %s here %s there %s more", new int[]{0,1}, new double[]{2.0, 3.0}, new boolean[]{true, false});
        } catch (NullPointerException e){
            assertEquals("Message [0, 1] here [2.0, 3.0] there [true, false] more", e.getMessage());
        }

        Preconditions.checkNotNull("", "Message %s here %s there", new String[]{"A", "B"}, new Object[]{1.0, "C"});
        try{
            Preconditions.checkNotNull(null, "Message %s here %s there", new String[]{"A", "B"}, new Object[]{1.0, "C"});
        } catch (NullPointerException e){
            assertEquals("Message [A, B] here [1.0, C] there", e.getMessage());
        }
    }

}
