echo Start

cd my_tool
mkdir /tmp/salt-tests-tmpdir
python -u test_main.py -d /home/wonseok/benchmark -c p > ./result/github_result.log
echo my_tool Done 
cd ../no_typeaware
mkdir /tmp/salt-tests-tmpdir
python -u test_main.py -d /home/wonseok/benchmark -c p > ./result/github_result.log
echo no_typeaware Done 
cd ../no_static
mkdir /tmp/salt-tests-tmpdir
python -u test_main.py -d /home/wonseok/benchmark -c p > ./result/github_result.log
echo no_static Done 
cd ../baseline_final
mkdir /tmp/salt-tests-tmpdir
python -u test_main.py -d /home/wonseok/benchmark -c p > ./result/github_result.log
cd ..
echo baseline Done 