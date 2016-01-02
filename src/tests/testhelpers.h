/*
 * testhelpers.h
 *
 *  Created on: Jan 1, 2016
 *      Author: agibsonccc
 */

#ifndef TESTHELPERS_H_
#define TESTHELPERS_H_
int arrsEquals(int rank,int *comp1,int *comp2);


int arrsEquals(int rank, int *comp1,int *comp2) {
	int ret = 1;
	for(int i = 0; i < rank; i++) {
		ret = ret && (comp1[i] == comp2[i]);
	}

	return ret;
}

#endif /* TESTHELPERS_H_ */
