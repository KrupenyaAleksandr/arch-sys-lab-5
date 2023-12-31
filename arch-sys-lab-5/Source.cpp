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
	#pragma omp parallel sections
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
	std::cout << "\n�������� ���� ��������, ���������� ��������� - " << count << std::endl;
	std::cout << "�����: " << sum << ", �����: " << omp_get_wtime() - timer1 << "\n\n";
}

void printArray(int* array, int size) {
	for (int i = 0; i < size; ++i) {
		std::cout << array[i] << " ";
	}
	std::cout << std::endl;
}

template <typename T>
void randomArray(T* array, int size) {
	for (int i = 0; i < size; ++i)
		array[i] = rand() % 10000000;
}

template <typename T>
void randomVec(std::vector<T>& vec, int size) {
	for (int i = 0; i < size; ++i)
		vec.push_back(rand() % 1000);
}

void isSorted(int* array, int size) {
	for (int i = 0; i < size - 1; ++i) {
		if (array[i] > array[i + 1])
			std::cout << "Not sorted";
	}
	std::cout << "Sorted";
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

void mergeSortParallel(int* srcArray, int n, int* dstArray, int min_partition) {
	if (n < 2)
		return;
	if (n > min_partition) {
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				mergeSortParallel(srcArray, n / 2, dstArray, min_partition);
			}
			#pragma omp section
			{
				mergeSortParallel(srcArray + (n / 2), n - (n / 2), dstArray + n / 2, min_partition);
			}
		}
	}
	else {
		mergeSortParallel(srcArray, n / 2, dstArray, min_partition);
		mergeSortParallel(srcArray + (n / 2), n - (n / 2), dstArray + n / 2, min_partition);
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

void task2(int* srcArray, long int size) {
	int* tmpArray = new int[size];
	copyArray(srcArray, tmpArray, size);
	int* dstArray = new int[size];
	std::cout << "���������� ��������, ���������� ��������� - " << size << std::endl;

	auto timer1 = omp_get_wtime();
	mergeSortRegular(tmpArray, size, dstArray);
	std::cout << "������� ���������� ���������" << std::endl;
	std::cout << "�����: " << omp_get_wtime() - timer1 << std::endl;

	delete[] tmpArray;
	delete[] dstArray;
	dstArray = new int[size];
	tmpArray = new int[size];
	copyArray(srcArray, tmpArray, size);

	omp_set_num_threads(2);
	timer1 = omp_get_wtime();
	mergeSortParallel(tmpArray, size, dstArray, size * 5 / 100);
	std::cout << "������������ ���������� ��������� ��� ���������� ������������" << std::endl;
	std::cout << "�����: " << omp_get_wtime() - timer1 << std::endl;

	delete[] tmpArray;
	delete[] dstArray;
	dstArray = new int[size];
	tmpArray = new int[size];
	copyArray(srcArray, tmpArray, size);

	omp_set_nested(1);
	timer1 = omp_get_wtime();
	mergeSortParallel(tmpArray, size, dstArray, size * 5 / 100);
	std::cout << "������������ ���������� ��������� �� ��������� �������������" << std::endl;
	std::cout << "�����: " << omp_get_wtime() - timer1 << "\n\n"; 
	omp_set_nested(0);
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


void quickSortParallel(int* array, int start, int end, int min_partition) {
	if (start >= end)
		return;

	int supElem = quickPartition(array, start, end);

	if (end - start > min_partition) {
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				quickSortParallel(array, start, supElem - 1, min_partition);
			}
			#pragma omp section
			{
				quickSortParallel(array, supElem + 1, end, min_partition);
			}
		}
	}
	else {
		quickSortParallel(array, start, supElem - 1, min_partition);
		quickSortParallel(array, supElem + 1, end, min_partition);
	}
}

void task3(int* srcArray, long int size) {
	int* dstArray = new int[size];
	copyArray(srcArray, dstArray, size);
	std::cout << "������� ����������, ���������� ��������� - " << size << std::endl;

	auto timer1 = omp_get_wtime();
	quickSortRegular(dstArray, 0, size - 1);
	std::cout << "������� ������� ����������" << std::endl;
	std::cout << "�����: " << omp_get_wtime() - timer1 << std::endl;

	delete[] dstArray;
	dstArray = new int[size];
	copyArray(srcArray, dstArray, size);

    omp_set_num_threads(2);
	timer1 = omp_get_wtime();
	quickSortParallel(dstArray, 0, size - 1, size * 5 / 100);
	std::cout << "������������ ������� ���������� ��� ���������� ������������" << std::endl;
	std::cout << "�����: " << omp_get_wtime() - timer1 << std::endl;

	delete[] dstArray;
	dstArray = new int[size];
	copyArray(srcArray, dstArray, size);

	omp_set_nested(1);
    timer1 = omp_get_wtime();
	quickSortParallel(dstArray, 0, size - 1, size * 5 / 100);
    std::cout << "������������ ������� ���������� �� ��������� �������������" << std::endl;
	std::cout << "�����: " << omp_get_wtime() - timer1 << "\n\n";
	omp_set_nested(0);
}

#ifdef _OPENMP
int main() {
	setlocale(LC_ALL, "");
	srand(time(NULL));
	long int size;
	std::cout << "������� ���������� ��������� � �������: ";
	std::cin >> size;
	int* srcArray = new int[size];
	randomArray(srcArray, size);

	std::vector<double> vec1;
	randomVec(vec1, 1000000);
	std::vector<int> vec2;
	randomVec(vec2, 1000000);
	task1(vec1, vec2);
	task2(srcArray, size);
	task3(srcArray, size);
	return 1;
}
#endif