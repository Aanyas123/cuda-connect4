# Build with: make ARCH=sm_86   (the Coursera lab's nvcc has no sm_89; sm_86
# binaries run on its compute-8.9 L4 GPU)
NVCC  ?= nvcc
ARCH  ?= sm_86
FLAGS := -O2 -std=c++14 -arch=$(ARCH) -Xcompiler -Wall

TARGET := bin/connect4
SRCS   := src/main.cu src/strategies.cu src/board_util.cc src/game_file.cc
HDRS   := src/board.h src/board_util.h src/cuda_check.cuh src/game_file.h \
          src/strategies.cuh

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	@mkdir -p bin
	$(NVCC) $(FLAGS) -o $@ $(SRCS)

run: $(TARGET)
	./run.sh $(ARCH)

clean:
	rm -rf bin games

.PHONY: all run clean
