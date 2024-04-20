MSG=update

clean:
	rm -fr build && rm -fr bin && rm -f output.bin

debug-build: clean
	mkdir -p bin && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j32

debug: debug-build
	cd bin && gdb main

build: clean
	mkdir -p bin && mkdir build && cd build && cmake .. && make -j32

run:
	cd bin && ./main

run-1m:
	cd bin && ./main ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin

run-10m:
	cd bin && ./main ../data/contest-data-release-10m.bin ../data/contest-queries-release-10m.bin

run-recall-1m:
	cd bin && ./test_recall

git-push:
	git add . && git commit -m "$(MSG)" && git push origin main