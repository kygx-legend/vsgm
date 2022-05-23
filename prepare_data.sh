#!/usr/bin/env bash

export VSGM_HOME=.
export VSGM_PREPROCESS=${VSGM_HOME}/preprocess_graph
export VSGM_FEATURES=${VSGM_HOME}/features
export VSGM_KMEANS=${VSGM_HOME}/kmeans
export VSGM_BIN_PACKING=${VSGM_HOME}/view_packing


export DATA_DIR=${VSGM_HOME}/data

# mkdir data
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[Run] mkdir ${DATA_DIR}"
  mkdir ${DATA_DIR}
  echo "[Done] mkdir ${DATA_DIR}"
fi

# download
if [[ ! -f "${DATA_DIR}/com-friendster.ungraph.txt.gz" ]] && [[ ! -f "${DATA_DIR}/com-friendster.ungraph.txt" ]]; then
  echo "[Run] download friendster graph to ${DATA_DIR}"
  cd ${DATA_DIR} && wget https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz && gunzip com-friendster.ungraph.txt.gz && cd ..
  echo "[Done] download friendster graph to ${DATA_DIR}"
fi

# preprocess
if [[ -f "${DATA_DIR}/com-friendster.ungraph.txt" ]]; then
  echo "[Run] preprocess friendster graph"
  ${VSGM_PREPROCESS} -f ${DATA_DIR}/com-friendster.ungraph.txt -d 0 -o 1
  echo "[Done] preprocess friendster graph"
fi

# features
if [[ -f "${DATA_DIR}/com-friendster.ungraph.txt.bin" ]]; then
  echo "[Run] get features on friendster graph"
  ${VSGM_FEATURES} -f ${DATA_DIR}/com-friendster.ungraph.txt.bin -h 1
  echo "[Done] get features on friendster graph"
fi

# kmeans
if [[ -f "${DATA_DIR}/com-friendster.ungraph.txt.bin.1hop.bin" ]]; then
  echo "[Run] run kmeans on friendster graph"
  ${VSGM_KMEANS} -f ${DATA_DIR}/com-friendster.ungraph.txt.bin -i 10 -t 32 -k 4 -init 0 -rc 1
  echo "[Done] run kmeans on friendster graph"
fi

# view bin packing
if [[ -f "${DATA_DIR}/com-friendster.ungraph.txt.bin.kmeans.4" ]]; then
  echo "[Run] run view bin packing on friendster graph"
  cp ${DATA_DIR}/com-friendster.ungraph.txt.bin.kmeans.4 ${DATA_DIR}/com-friendster.ungraph.txt.bin.kmeans.1x4
  ${VSGM_BIN_PACKING} -gf ${DATA_DIR}/com-friendster.ungraph.txt.bin -pf ${DATA_DIR}/com-friendster.ungraph.txt.bin.kmeans.1x4 -h 2 -m 10 -t 1 -s 1 -d 1
  echo "[Done] run view bin packing on friendster graph"
fi
