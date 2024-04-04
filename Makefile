MSG=update

clean:
	rm -fr build && rm -fr bin && rm output.bin || true

debug-build: clean
	mkdir -p bin && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j

debug: debug-build
	cd bin && gdb main

build: clean
	mkdir -p bin && mkdir build && cd build && cmake .. && make -j

run:
	cd bin && ./main

git-push:
	git add . && git commit -m "$(MSG)" && git push origin main