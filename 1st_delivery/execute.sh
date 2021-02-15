#!/bin/bash
if [ $1 -eq 1 ]; then
  python src/rule-based/Goal1/nerc.py
elif [ $1 -eq 2 ]; then
  python src/rule-based/Goal2/nerc.py
elif [ $1 -eq 3 ]; then
  python src/ml/Goal1/nerc.py
elif [ $1 -eq 4 ]; then
  python src/ml/Goal2/nerc.py
else
  echo "Something went wrong :("
fi
