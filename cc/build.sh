c++ -O4 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) mpc.cc -o mpc$(python3-config --extension-suffix)
cp *.so ../testbed/
mv *.so ../
echo 'Done'