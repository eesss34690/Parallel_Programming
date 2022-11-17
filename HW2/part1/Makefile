FINALTARGET := pi.out
CPP := pi.cpp

CXX := g++
CFLAGS := -std=c++20 -Ofast
CFLAGS += -ffast-math -mfma -m64 -flto -fPIC -pthread

all: $(CPP)
	$(CXX) $(CPP) -o $(FINALTARGET) $(CFLAGS) $(THREAD)
 
clean:
	rm  *.out
