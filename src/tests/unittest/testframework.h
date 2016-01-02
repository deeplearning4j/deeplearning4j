/*
 * testframework.h
 *
 *  Created on: Dec 29, 2015
 *      Author: agibsonccc
 */

#ifndef TESTFRAMEWORK_H_
#define TESTFRAMEWORK_H_


#pragma once

#include <string>
#include <vector>
#include <set>
#include <map>
#include <iostream>

#include <stdio.h>

#include "meta.h"
#include "util.h"

// define some common lists of types
typedef unittest::type_list<int,
		unsigned int,
		float> ThirtyTwoBitTypes;

typedef unittest::type_list<long long,
		unsigned long long,
		double> SixtyFourBitTypes;

typedef unittest::type_list<char,
		signed char,
		unsigned char,
		short,
		unsigned short,
		int,
		unsigned int,
		long,
		unsigned long,
		long long,
		unsigned long long> IntegralTypes;

typedef unittest::type_list<signed char,
		signed short,
		signed int,
		signed long,
		signed long long> SignedIntegralTypes;

typedef unittest::type_list<unsigned char,
		unsigned short,
		unsigned int,
		unsigned long,
		unsigned long long> UnsignedIntegralTypes;

typedef unittest::type_list<char,
		signed char,
		unsigned char> ByteTypes;

typedef unittest::type_list<char,
		signed char,
		unsigned char,
		short,
		unsigned short> SmallIntegralTypes;

typedef unittest::type_list<long long,
		unsigned long long> LargeIntegralTypes;

typedef unittest::type_list<float,
		double> FloatingPointTypes;

typedef unittest::type_list<char,
		signed char,
		unsigned char,
		short,
		unsigned short,
		int,
		unsigned int,
		long,
		unsigned long,
		long long,
		unsigned long long,
		float> NumericTypes;
// exclude double from NumericTypes


inline void chop_prefix(std::string& str, const std::string& prefix)
{
	str.replace(str.find(prefix) == 0 ? 0 : str.size(), prefix.size(), "");
}

inline std::string base_class_name(const std::string& name)
{
	std::string result = name;

	// if the name begins with "struct ", chop it off
	chop_prefix(result, "struct ");

	// if the name begins with "class ", chop it off
	chop_prefix(result, "class ");

	// chop everything including and after first "<"
	return result.replace(result.find_first_of("<"),
			result.size(),
			"");
}

enum TestStatus { Pass = 0, Failure = 1, KnownFailure = 2, Error = 3, UnknownException = 4};

typedef std::set<std::string>              ArgumentSet;
typedef std::map<std::string, std::string> ArgumentMap;

std::vector<size_t> get_test_sizes(void);
void                set_test_sizes(const std::string&);

class UnitTest {
public:
	std::string name;
	UnitTest() {}
	UnitTest(const std::string name);
	virtual ~UnitTest() {}
	virtual void run() {}

	bool operator<(const UnitTest& u) const
	{
		return name < u.name;
	}
};

class UnitTestDriver;

class UnitTestDriver
{
	typedef std::map<std::string, UnitTest*> TestMap;
	typedef std::pair<std::string, UnitTest*> TestPair;
	TestMap test_map;

	bool run_tests(std::vector<UnitTest *>& tests_to_run, const ArgumentMap& kwargs);

protected:
	// executed immediately after each test
	// \param test The UnitTest of interest
	// \param concise Whether or not to suppress output
	// \return true if all is well; false if the tests must be immediately aborted
	virtual bool post_test_sanity_check(const UnitTest &test, bool concise);

public:
	inline virtual ~UnitTestDriver() {};

	void register_test(UnitTest * test);
	virtual bool run_tests(const ArgumentSet& args, const ArgumentMap& kwargs);
	void list_tests(void);


};


// Macro to create a single unittest
#define DECLARE_UNITTEST(TEST)                                   \
		class TEST##UnitTest : public UnitTest {                         \
		public:                                                      \
		TEST##UnitTest() : UnitTest(#TEST) {}                        \
		void run(){                                                  \
			TEST();                                              \
		}                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

// Macro to create host and device versions of a
// unit test for a couple data types
#define DECLARE_VECTOR_UNITTEST(VTEST)                                                                            \
		void VTEST##Host(void)   {  VTEST< thrust::host_vector<short> >();   VTEST< thrust::host_vector<int> >();   }    \
		void VTEST##Device(void) {  VTEST< thrust::device_vector<short> >(); VTEST< thrust::device_vector<int> >(); }    \
		DECLARE_UNITTEST(VTEST##Host);                                                                                    \
		DECLARE_UNITTEST(VTEST##Device);

// Macro to create instances of a test for several
// data types and array sizes
#define DECLARE_VARIABLE_UNITTEST(TEST)                          \
		class TEST##UnitTest : public UnitTest {                         \
		public:                                                      \
		TEST##UnitTest() : UnitTest(#TEST) {}                        \
		void run()                                                   \
		{                                                            \
			std::vector<size_t> sizes = get_test_sizes();            \
			for(size_t i = 0; i != sizes.size(); ++i)                \
			{                                                        \
				TEST<char>(sizes[i]);                                \
				TEST<unsigned char>(sizes[i]);                       \
				TEST<short>(sizes[i]);                               \
				TEST<unsigned short>(sizes[i]);                      \
				TEST<int>(sizes[i]);                                 \
				TEST<unsigned int>(sizes[i]);                        \
				TEST<float>(sizes[i]);                               \
			}                                                        \
		}                                                            \
};                                                               \
TEST##UnitTest TEST##Instance

template<template <typename> class TestName, typename TypeList>
class SimpleUnitTest : public UnitTest
{
public:
	SimpleUnitTest()
: UnitTest(base_class_name(unittest::type_name<TestName<int> >()).c_str()) {}

	void run()
	{
		// get the first type in the list
		typedef typename unittest::get_type<TypeList,0>::type first_type;

		unittest::for_each_type<TypeList,TestName,first_type,0> for_each;

		// loop over the types
		for_each();
	}
}; // end SimpleUnitTest


template<template <typename> class TestName, typename TypeList>
class VariableUnitTest : public UnitTest
{
public:
	VariableUnitTest()
: UnitTest(base_class_name(unittest::type_name<TestName<int> >()).c_str()) {}

	void run()
	{
		std::vector<size_t> sizes = get_test_sizes();
		for(size_t i = 0; i != sizes.size(); ++i)
		{
			// get the first type in the list
			typedef typename unittest::get_type<TypeList,0>::type first_type;

			unittest::for_each_type<TypeList,TestName,first_type,0> loop;

			// loop over the types
			loop(sizes[i]);
		}
	}
}; // end VariableUnitTest

template<template <typename> class TestName,
typename TypeList,
template <typename, typename> class Vector,
template <typename> class Alloc>
struct VectorUnitTest
		: public UnitTest
		  {
	VectorUnitTest()
	: UnitTest((base_class_name(unittest::type_name<TestName< Vector<int, Alloc<int> > > >()) + "<" +
			base_class_name(unittest::type_name<Vector<int, Alloc<int> > >()) + ">").c_str())
	{ }

	void run()
	{
		// zip up the type list with Alloc
		typedef typename unittest::transform1<TypeList, Alloc>::type AllocList;

		// zip up the type list & alloc list with Vector
		typedef typename unittest::transform2<TypeList, AllocList, Vector>::type VectorList;

		// get the first type in the list
		typedef typename unittest::get_type<VectorList,0>::type first_type;

		unittest::for_each_type<VectorList,TestName,first_type,0> loop;

		// loop over the types
		loop(0);
	}
		  }; // end VectorUnitTest



#endif /* TESTFRAMEWORK_H_ */
