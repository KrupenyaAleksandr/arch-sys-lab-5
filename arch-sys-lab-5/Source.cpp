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
	#pragma omp parallel sections num_threads(2)
	{
		#pragma omp section
		{
			mergeSortParallelNested(srcArray, n / 2, dstArray);
		}
		#pragma omp section
		{
			mergeSortParallelNested(srcArray + (n / 2), n - (n / 2), dstArray + n / 2);
		}
	}
	mergeFusion(srcArray, n, dstArray);
}

void mergeSortParallel(int* srcArray, int n, int* dstArray) {
	if (n < 2)
		return;
	#pragma omp parallel sections num_threads(2)
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
	long int size = 100000;
	int* srcArray = new int[size];
	randomArray(srcArray, size);
	//printArray(srcArray, size);
	int* dstArray = new int[size];
	auto timer1 = omp_get_wtime();
	mergeSortRegular(srcArray, size, dstArray);
	std::cout << omp_get_wtime() - timer1 << std::endl;

	//delete[] srcArray;
	delete[] dstArray;
	//srcArray = new int[size];
	//randomArray(srcArray, size);
	dstArray = new int[size];
	//std::cout << omp_in_parallel();
	timer1 = omp_get_wtime();
	//omp_set_nested(1);
	#pragma omp parallel
	{
		#pragma omp single
		{
			mergeSortParallel(srcArray, size, dstArray);
		}
	}
	//printArray(dstArray, size);
	std::cout << omp_get_wtime() - timer1 << std::endl;

	//delete[] srcArray;
	delete[] dstArray;
	//srcArray = new int[size];
	//randomArray(srcArray, size);
	dstArray = new int[size];
	timer1 = omp_get_wtime();
	//omp_set_nested(1);
	//#pragma omp parallel
	//{
		//#pragma omp single
		//{
			mergeSortParallelNested(srcArray, size, dstArray);
		//}
	//}
	std::cout << omp_get_wtime() - timer1 << std::endl;
}



void task3() {

}

#ifdef _OPENMP
int main() {
	srand(time(NULL));
	std::vector<double> vec1 = { 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
	std::vector<int> vec2 = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29 };
	task1(vec1, vec2);

	task2();
	return 1;
}
#endif