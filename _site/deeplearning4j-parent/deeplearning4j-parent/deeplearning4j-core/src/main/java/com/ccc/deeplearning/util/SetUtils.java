package com.ccc.deeplearning.util;

import java.util.Collection;
import java.util.HashSet ;
import java.util.Set ;

public class SetUtils
{
	private SetUtils() {}

	// Set specific operations

	public static <T> Set<T> intersection(Collection<T> parentCollection, Collection<T> removeFromCollection)
	{
		Set<T> results = new HashSet<T>(parentCollection) ;
		results.retainAll(removeFromCollection) ;
		return results ;
	}

	public static <T> boolean intersectionP(Set<? extends T> s1, Set<? extends T> s2)
	{
		for( T elt : s1 )
		{
			if ( s2.contains(elt) ) 
				return true ;
		}
		return false ;
	}

	public static <T> Set<T> union(Set<? extends T> s1, Set<? extends T> s2)
	{
		Set<T> s3 = new HashSet<T>(s1) ;
		s3.addAll(s2) ;
		return s3 ;
	}

	/** Return is s1 \ s2 */

	public static <T> Set<T> difference(Collection<? extends T> s1, Collection<? extends T> s2)
	{
		Set<T> s3 = new HashSet<T>(s1) ;
		s3.removeAll(s2) ;
		return s3 ;
	}
}


/*
 * (c) Copyright 2006, 2007, 2008, 2009 Hewlett-Packard Development Company, LP
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

