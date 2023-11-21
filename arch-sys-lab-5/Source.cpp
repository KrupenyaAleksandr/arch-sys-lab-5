#include <iostream>
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

template <typename T1, typename T2>
void task1(std::vector <T1> vec1, std::vector <T2> vec2) {
	double sum = 0;
	int count = vec1.size();
	auto timer1 = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp sections reduction(+:sum)
		{
			#pragma omp section
			{
				for (int i = 0; i < count / 2; ++i)
					sum += vec1[i] + vec2[i];
			}
			#pragma omp section
			{
				for (int i = count / 2; i < count; ++i)
					sum += vec1[i] + vec2[i];
			}
		}
	}
	std::cout << sum << " " << omp_get_wtime() - timer1 << std::endl;
}

void printArray(int* array, int size) {
	for (int i = 0; i < size; ++i) {
		std::cout << array[i] << " ";
	}
	std::cout << std::endl;
}

void randomArray(int* array, int size) {
	for (int i = 0; i < size; ++i)
		array[i] = rand() % 10000000;
}

bool isSorted(int* array, int size) {
	for (int i = 0; i < size - 1; ++i) {
		if (array[i] > array[i + 1]) 
			return false;
	}
	return true;
}

void copyArray(int* srcArray, int* dstArray, int size) {
	for (int i = 0; i < size; ++i) {
		dstArray[i] = srcArray[i];
	}
}

void mergeFusion(int* srcArray, int n, int* dstArray) {
	int left = 0;
	int right = n / 2;
	int pointer = 0;

	while (left < n / 2 && right < n) {
		if (srcArray[left] < srcArray[right]) {
			dstArray[pointer] = srcArray[left];
			pointer++;
			left++;
		}
		else {
			dstArray[pointer] = srcArray[right];
			pointer++;
			right++;
		}
	}
	while (left < n / 2) {
		dstArray[pointer] = srcArray[left];
		pointer++;
		left++;
	}
	while (right < n) {
		dstArray[pointer] = srcArray[right];
		pointer++;
		right++;
	}
	memcpy(srcArray, dstArray, n * sizeof(int));
}

void mergeSortParallelNested(int* srcArray, int n, int* dstArray) {
	if (n < 2)
		return;
	mergeSortParallelNested(srcArray, n / 2, dstArray);
	mergeSortParallelNested(srcArray + (n / 2), n - (n / 2), dstArray + n / 2);
	mergeFusion(srcArray, n, dstArray);
}

void mergeSortParallel(int* srcArray, int n, int* dstArray) {
	if (n < 2)
		return;
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			mergeSortParallel(srcArray, n / 2, dstArray);
		}
		#pragma omp section
		{
			mergeSortParallel(srcArray + (n / 2), n - (n / 2), dstArray + n / 2);
		}
	}
	mergeFusion(srcArray, n, dstArray);
}

void mergeSortRegular(int* srcArray, int n, int* dstArray) {
	if (n < 2)
		return;
	mergeSortRegular(srcArray, n / 2, dstArray);
	mergeSortRegular(srcArray + (n / 2), n - (n / 2), dstArray + n / 2);
	mergeFusion(srcArray, n, dstArray);
}

void task2() {
	long int size = 10000000;
	int* srcArray = new int[size];
	randomArray(srcArray, size);
	int* dstArray = new int[size];
	auto timer1 = omp_get_wtime();
	mergeSortRegular(srcArray, size, dstArray);
	std::cout << omp_get_wtime() - timer1 << std::endl;

	delete[] dstArray;

	dstArray = new int[size];
	timer1 = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp single
		{
			mergeSortParallel(srcArray, size, dstArray);
		}
	}
	std::cout << omp_get_wtime() - timer1 << std::endl;

	delete[] dstArray;

	dstArray = new int[size];
	timer1 = omp_get_wtime();
	omp_set_nested(1);
	mergeSortParallelNested(srcArray, size, dstArray);
	std::cout << omp_get_wtime() - timer1 << std::endl;
}

int quickPartition(int* array, int start, int end) {
	int supElem = array[end];
	int pIndex = start;

	for (int i = start; i < end; ++i) {
		if (array[i] <= supElem) {
			std::swap(array[i], array[pIndex]);
			pIndex++;
		}
	}

	std::swap(array[pIndex], array[end]);
	return pIndex;
}

void quickSortRegular(int* array, int start, int end) {
	if (start >= end)
		return;

	int supElem = quickPartition(array, start, end);

	quickSortRegular(array, start, supElem - 1);
	quickSortRegular(array, supElem + 1, end);
}


void quickSortParallel(int* array, int start, int end) {
	if (start >= end)
		return;

	int supElem = quickPartition(array, start, end);

	#pragma omp parallel sections
	{
		#pragma omp section
		{
			quickSortParallel(array, start, supElem - 1);
		}
		#pragma omp section
		{
			quickSortParallel(array, supElem + 1, end);
		}
	}
}

void quickSortParallelNested(int* array, int start, int end) {
	if (start >= end)
		return;

	int supElem = quickPartition(array, start, end);

	quickSortParallelNested(array, start, supElem - 1);
	quickSortParallelNested(array, supElem + 1, end);
}

void task3() {
	int size = 3000;
	int* srcArray = new int[size];
	randomArray(srcArray, size);
	int* dstArray = new int[size];
	copyArray(srcArray, dstArray, size);

	quickSortRegular(dstArray, 0, size - 1);
	//std::cout << std::endl << isSorted(dstArray, size);
	std::cout << std::endl;

	delete[] dstArray;
	dstArray = new int[size];
	copyArray(srcArray, dstArray, size);

	quickSortParallel(dstArray, 0, size - 1);
	//std::cout << std::endl << isSorted(dstArray, size);

	omp_set_nested(1);
}

#ifdef _OPENMP
int main() {
	setlocale(LC_ALL, "");
	srand(time(NULL));
	std::vector<double> vec1 = { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
	std::vector<int> vec2 = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29 };
	//task1(vec1, vec2);

	//task2();
	task3();
	return 1;
}
#endif