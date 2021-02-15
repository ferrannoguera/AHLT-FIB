#!/bin/bash
if [ $1 -eq 1 ]; then
  python src/rule-based/ddi.py
elif [ $1 -eq 2 ]; then
  echo "CREATING FEATURES"
  python src/ml/create_features.py
  echo "TRAINING"
  python src/ml/train_predict.py
else
  echo "Something went wrong :("
fi
