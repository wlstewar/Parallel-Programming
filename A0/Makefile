CXX=g++
CXXFLAGS=-fopenmp -std=c++17 -O3 -Wall -Wextra

.PHONY: clean all

all: clean a0

clean:
	rm -rf a0 A0.tar
archive:
	rm -f A0.tar
	tar --exclude='ccr' -cf A0.tar *
