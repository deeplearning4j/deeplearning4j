/*
 * meta.h
 *
 *  Created on: Dec 29, 2015
 *      Author: agibsonccc
 */

#ifndef META_H_
#define META_H_
/*! \file meta.h
 *  \brief Defines template classes
 *         for metaprogramming in the
 *         unit tests.
 */

#pragma once

namespace unittest
{

// mark the absence of a type
struct null_type {};

// this type encapsulates a list of
// up to 10 types
template<typename T0 = null_type,
         typename T1 = null_type,
         typename T2 = null_type,
         typename T3 = null_type,
         typename T4 = null_type,
         typename T5 = null_type,
         typename T6 = null_type,
         typename T7 = null_type,
         typename T8 = null_type,
         typename T9 = null_type,
         typename T10 = null_type,
         typename T11 = null_type,
         typename T12 = null_type,
         typename T13 = null_type,
         typename T14 = null_type,
         typename T15 = null_type,
         typename T16 = null_type,
         typename T17 = null_type,
         typename T18 = null_type,
         typename T19 = null_type>
  struct type_list
{
  typedef T0 type_0;
  typedef T1 type_1;
  typedef T2 type_2;
  typedef T3 type_3;
  typedef T4 type_4;
  typedef T5 type_5;
  typedef T6 type_6;
  typedef T7 type_7;
  typedef T8 type_8;
  typedef T9 type_9;
  typedef T10 type_10;
  typedef T11 type_11;
  typedef T12 type_12;
  typedef T13 type_13;
  typedef T14 type_14;
  typedef T15 type_15;
  typedef T16 type_16;
  typedef T17 type_17;
  typedef T18 type_18;
  typedef T19 type_19;
};

// this type provides a way of indexing
// into a type_list
template<typename List, unsigned int i>
  struct get_type
{
  typedef null_type type;
};

template<typename List>  struct get_type<List,0> { typedef typename List::type_0 type; };
template<typename List>  struct get_type<List,1> { typedef typename List::type_1 type; };
template<typename List>  struct get_type<List,2> { typedef typename List::type_2 type; };
template<typename List>  struct get_type<List,3> { typedef typename List::type_3 type; };
template<typename List>  struct get_type<List,4> { typedef typename List::type_4 type; };
template<typename List>  struct get_type<List,5> { typedef typename List::type_5 type; };
template<typename List>  struct get_type<List,6> { typedef typename List::type_6 type; };
template<typename List>  struct get_type<List,7> { typedef typename List::type_7 type; };
template<typename List>  struct get_type<List,8> { typedef typename List::type_8 type; };
template<typename List>  struct get_type<List,9> { typedef typename List::type_9 type; };
template<typename List>  struct get_type<List,10> { typedef typename List::type_10 type; };
template<typename List>  struct get_type<List,11> { typedef typename List::type_11 type; };
template<typename List>  struct get_type<List,12> { typedef typename List::type_12 type; };
template<typename List>  struct get_type<List,13> { typedef typename List::type_13 type; };
template<typename List>  struct get_type<List,14> { typedef typename List::type_14 type; };
template<typename List>  struct get_type<List,15> { typedef typename List::type_15 type; };
template<typename List>  struct get_type<List,16> { typedef typename List::type_16 type; };
template<typename List>  struct get_type<List,17> { typedef typename List::type_17 type; };
template<typename List>  struct get_type<List,18> { typedef typename List::type_18 type; };
template<typename List>  struct get_type<List,19> { typedef typename List::type_19 type; };

// this type and its specialization provides a way to
// iterate over a type_list, and
// applying a unary function to each type
template<typename TypeList,
         template <typename> class Function,
         typename T,
         unsigned int i = 0>
  struct for_each_type
{
  template<typename U>
    void operator()(U n)
  {
    // run the function on type T
    Function<T> f;
    f(n);

    // get the next type
    typedef typename get_type<TypeList,i+1>::type next_type;

    // recurse to i + 1
    for_each_type<TypeList, Function, next_type, i + 1> loop;
    loop(n);
  }

  void operator()(void)
  {
    // run the function on type T
    Function<T> f;
    f();

    // get the next type
    typedef typename get_type<TypeList,i+1>::type next_type;

    // recurse to i + 1
    for_each_type<TypeList, Function, next_type, i + 1> loop;
    loop();
  }
};

// terminal case: do nothing when encountering null_type
template<typename TypeList,
         template <typename> class Function,
         unsigned int i>
  struct for_each_type<TypeList, Function, null_type, i>
{
  template<typename U>
    void operator()(U n)
  {
    // no-op
  }

  void operator()(void)
  {
    // no-op
  }
};

// this type and its specialization instantiates
// a template by applying T to Template.
// if T == null_type, then its result is also null_type
template<template <typename> class Template,
         typename T>
  struct ApplyTemplate1
{
  typedef Template<T> type;
};

template<template <typename> class Template>
  struct ApplyTemplate1<Template, null_type>
{
  typedef null_type type;
};

// this type and its specializations instantiates
// a template by applying T1 & T2 to Template.
// if either T1 or T2 == null_type, then its result
// is also null_type
template<template <typename,typename> class Template,
         typename T1,
         typename T2>
  struct ApplyTemplate2
{
  typedef Template<T1,T2> type;
};

template<template <typename,typename> class Template,
         typename T>
  struct ApplyTemplate2<Template, T, null_type>
{
  typedef null_type type;
};

template<template <typename,typename> class Template,
         typename T>
  struct ApplyTemplate2<Template, null_type, T>
{
  typedef null_type type;
};

template<template <typename,typename> class Template>
  struct ApplyTemplate2<Template, null_type, null_type>
{
  typedef null_type type;
};

// this type creates a new type_list by applying a Template to each of
// the Type_list's types
template<typename TypeList,
         template <typename> class Template>
  struct transform1
{
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,0>::type>::type type_0;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,1>::type>::type type_1;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,2>::type>::type type_2;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,3>::type>::type type_3;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,4>::type>::type type_4;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,5>::type>::type type_5;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,6>::type>::type type_6;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,7>::type>::type type_7;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,8>::type>::type type_8;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,9>::type>::type type_9;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,10>::type>::type type_10;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,11>::type>::type type_11;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,12>::type>::type type_12;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,13>::type>::type type_13;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,14>::type>::type type_14;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,15>::type>::type type_15;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,16>::type>::type type_16;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,17>::type>::type type_17;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,18>::type>::type type_18;
  typedef typename ApplyTemplate1<Template, typename get_type<TypeList,19>::type>::type type_19;

  typedef type_list<type_0, type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8, type_9,
                    type_10, type_11, type_12, type_13, type_14, type_15, type_16, type_17, type_18, type_19> type;
};

// this type creates a new type_list by applying a Template to each of
// two type_list's types
template<typename TypeList1,
         typename TypeList2,
         template <typename,typename> class Template>
  struct transform2
{
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,0>::type, typename get_type<TypeList2,0>::type>::type type_0;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,1>::type, typename get_type<TypeList2,1>::type>::type type_1;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,2>::type, typename get_type<TypeList2,2>::type>::type type_2;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,3>::type, typename get_type<TypeList2,3>::type>::type type_3;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,4>::type, typename get_type<TypeList2,4>::type>::type type_4;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,5>::type, typename get_type<TypeList2,5>::type>::type type_5;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,6>::type, typename get_type<TypeList2,6>::type>::type type_6;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,7>::type, typename get_type<TypeList2,7>::type>::type type_7;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,8>::type, typename get_type<TypeList2,8>::type>::type type_8;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,9>::type, typename get_type<TypeList2,9>::type>::type type_9;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,10>::type, typename get_type<TypeList2,10>::type>::type type_10;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,11>::type, typename get_type<TypeList2,11>::type>::type type_11;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,12>::type, typename get_type<TypeList2,12>::type>::type type_12;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,13>::type, typename get_type<TypeList2,13>::type>::type type_13;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,14>::type, typename get_type<TypeList2,14>::type>::type type_14;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,15>::type, typename get_type<TypeList2,15>::type>::type type_15;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,16>::type, typename get_type<TypeList2,16>::type>::type type_16;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,17>::type, typename get_type<TypeList2,17>::type>::type type_17;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,18>::type, typename get_type<TypeList2,18>::type>::type type_18;
  typedef typename ApplyTemplate2<Template, typename get_type<TypeList1,19>::type, typename get_type<TypeList2,19>::type>::type type_19;


  typedef type_list<type_0, type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8, type_9,
                    type_10, type_11, type_12, type_13, type_14, type_15, type_16, type_17, type_18, type_19> type;
};

} // end unittest




#endif /* META_H_ */
