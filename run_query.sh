#!/usr/bin/env bash

get_query_arg() {
  query="0"
  if [ ${1} = "kc3" ]; then query="-kc 3"; fi
  if [ ${1} = "kc4" ]; then query="-kc 4"; fi
  if [ ${1} = "kc5" ]; then query="-kc 5"; fi
  if [ ${1} = "kc6" ]; then query="-kc 6"; fi
  if [ ${1} = "kc7" ]; then query="-kc 7"; fi
  if [ ${1} = "p1" ]; then query="-p 1"; fi
  if [ ${1} = "p2" ]; then query="-p 2"; fi
  if [ ${1} = "p3" ]; then query="-p 3"; fi
  if [ ${1} = "p4" ]; then query="-p 4"; fi
  if [ ${1} = "p5" ]; then query="-p 5"; fi
  if [ ${1} = "p6" ]; then query="-p 6"; fi
  if [ ${1} = "p7" ]; then query="-p 7"; fi
  if [ ${1} = "p8" ]; then query="-p 8"; fi
  if [ ${1} = "p9" ]; then query="-p 9"; fi
  if [ ${1} = "p10" ]; then query="-p 10"; fi
  if [ ${1} = "p11" ]; then query="-p 11"; fi
  if [ ${1} = "p12" ]; then query="-p 12"; fi
  # p13 in paper => p18 in code
  if [ ${1} = "p13" ]; then query="-p 18"; fi
	echo $query
}

run() {
  if [ $# -lt 4 ]; then echo 'run <vsgm_executable> <filename> <max_bin_size> <query>'; return; fi
  VSGM=$1
  file_name=$2
  mem=$3
  query=`get_query_arg ${4}`

  cmd="./${VSGM} -f $file_name -t 32 -m $mem -dm 1 -qs 3 -pn 3 -cn 1 $query"
  echo $cmd
  $cmd > $4.log 2>&1
}

run_all() {
  if [ $# -lt 3 ]; then echo 'run_all <vsgm_executable> <filename> <max_bin_size>'; return; fi
  VSGM=$1
  file_name=$2
  mem=$3
  for i in {3..7}; do run $VSGM $file_name $mem kc$i; done
  for i in {1..10}; do run $VSGM $file_name $mem p$i; done
}

run_multi() {
  if [ $# -lt 6 ]; then echo 'run_multi <vsgm_executable> <filename> <max_bin_size> <producer_num> <device_num> <query>'; return; fi
  VSGM=$1
  file_name=$2
  mem=$3
  producer_num=${4}
  device_num=${5}
  query=`get_query_arg ${6}`

  cmd="./${VSGM} -f $file_name -t 32 -m $mem -dm 1 -qs $producer_num -pn $producer_num -cn $device_num $query -dr 1"
  echo $cmd
  $cmd > $6.log 2>&1
}

print_help() {
  echo Usage:
  echo -n '  ./run_query.sh '; run
}

if [ $# -gt 0 ]; then
  if [ $1 == "-h" ]; then
    print_help
    exit
  fi
  $@
else
  print_help
fi
