1. Add dependencies of Nova by adding
   `source /data/share/users/zzxx/repos/inst_scripts/instrc.sh`
   in your `~/.bashrc` and remember to `source ~/.bashrc`.
   To remove the dependencies, comment the added command and re-source ~/.bashrc.

2. Compile RecursiveKMeans as a CMake project.
   a. `mkdir release; cd release`
   b. `cmake .. -DCMAKE_BUILD_TYPE=Release`
   c. `make RecursiveKMeans`

3. Pass the arguments to RecursiveKMeans to run it.
