############################################################################################
# <summary> CMakeLists.txt for ECE408 								</summary>
# <date>    2021-06-07          											</date>
# <author>  yuanxiulong 			</author>
# <email>   yuanxl19@mails.tsinghua.edu.cn	</email>
############################################################################################
project(ECE408)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
find_package(CUDA REQUIRED)
# collect source files

if(CUDA_FOUND)
    # compared to class settings, we let NVidia's FindCUDA CMake detect 
    # whether to build x64.  We tell it to support most devices, though, 
    # to make sure more people can easily run class code without knowing 
    # about this compiler argument
    if (MSVC)
        set(CUDA_NVCC_FLAGS "")
    else (MSVC)
        set(CUDA_NVCC_FLAGS "-gencode;arch=compute_75,code=sm_75;")
    endif (MSVC)

    # add -Wextra compiler flag for gcc compilations
    if (UNIX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wall;")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    endif (UNIX)

    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
	include_directories(${CMAKE_SOURCE_DIR})
	include_directories(${CUDA_INCLUDE_DIRS})

else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()

file( GLOB  wbhdr lib/*.hpp lib/*.h )
file( GLOB  wbsrc lib/*.cpp lib/*.c )

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories("lib/")

# Add my implementations
CUDA_ADD_EXECUTABLE(MP0_Solution implementations/mp0_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP1_Solution implementations/mp1_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP2_Solution implementations/mp2_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP3_Solution implementations/mp3_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP4_Solution implementations/mp4_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP5_Solution_Naive implementations/mp5_solution_naive.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP5_Solution implementations/mp5_solution.cu ${wbhdr} ${wbsrc})


# Add the solutions code
CUDA_ADD_EXECUTABLE(MP1_Provided_Solution solutions/mp1_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP2_Provided_Solution solutions/mp2_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP3_Provided_Solution solutions/mp3_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP4_Provided_Solution solutions/mp4_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP5_Provided_Solution solutions/mp5_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP6_Provided_Solution solutions/mp6_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP7_Provided_Solution solutions/mp7_solution.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP8_Provided_Solution solutions/mp8_solution.cu ${wbhdr} ${wbsrc})
#Add Tools
CUDA_ADD_EXECUTABLE(MP1_Dataset tools/mp1_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP2_Dataset tools/mp2_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP3_Dataset tools/mp3_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP4_Dataset tools/mp4_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP5_Dataset tools/mp5_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP6_Dataset tools/mp6_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(MP7_Dataset tools/mp7_dataset.cpp ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(Bank_Conflict tools/bank_conflict.cu ${wbhdr} ${wbsrc})
CUDA_ADD_EXECUTABLE(Memory_Transfer tools/memory_transfer.cu ${wbhdr} ${wbsrc})
