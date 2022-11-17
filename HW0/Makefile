all:
	g++ pi.cpp -o pi.out

gprof:
	g++ -pg pi.cpp -o pi.out
gprof_test:
	./pi.out
	gprof ./pi.out gmon.out -b > profiling_result

perf:
	g++ -g pi.cpp -o pi.out
perf_test:
	sudo perf record ./pi.out
	perf report

clean:
	rm -rf pi.out