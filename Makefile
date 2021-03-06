NVFLAGS  := -std=c++11

CXXFLAGS := -fopenmp

LDFLAGS  := -lm

MPILIBS  := -I/opt/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/include -L/opt/intel/compilers_and_libraries_2017.3.191/linux/mpi/intel64/lib -lmpi

EXES     := mpitest HW4_cuda cuda_debug offset_test HW4_openmp HW4_mpi

alls: $(EXES)

clean:
	rm -f $(EXES)

mpitest: mpitest.cu
	nvcc $(NVFLAGS) $(MPILIBS) -Xcompiler="$(CXXFLAGS)" -o $@ $?

HW4_cuda: HW4_cuda.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?

cuda_debug: cuda_debug.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?

offset_test: offset_test.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?

HW4_openmp: HW4_openmp.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?

HW4_mpi: HW4_mpi.cu
	nvcc $(NVFLAGS) $(MPILIBS) -Xcompiler="$(CXXFLAGS)" -o $@ $?
