#!/bin/bash
# Load shared environment variables
source "$(dirname "$0")/env.sh"

cd ./../../

git clone https://github.com/airoa-org/hsr_openpi.git -b feature/group4
