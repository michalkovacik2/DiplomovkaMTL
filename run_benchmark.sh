#!/bin/bash
# This should be run in conda enviroment detectron2
echo 'Running multi task learning model'
python3 benchmark.py    \
  --batchSize=1         \
  --inputSize=32        \
  --backbone=resnet18   \
  --ageHead=1           \
  --genderHead=1        \
  --ethnicityHead=1     \
  --model='./models/resnet18/best.pth'

echo 'Running only age'
python3 benchmark.py    \
  --batchSize=1         \
  --inputSize=32        \
  --backbone=resnet18   \
  --ageHead=1           \
  --genderHead=0        \
  --ethnicityHead=0     \
  --model='./models/resnet18-age/best.pth'

echo 'Running gender only'
python3 benchmark.py    \
  --batchSize=1         \
  --inputSize=32        \
  --backbone=resnet18   \
  --ageHead=0           \
  --genderHead=1        \
  --ethnicityHead=0     \
  --model='./models/resnet18-gender/best.pth'

echo 'Running ethnicity only'
python3 benchmark.py   \
  --batchSize=1        \
  --inputSize=32       \
  --backbone=resnet18  \
  --ageHead=0          \
  --genderHead=0       \
  --ethnicityHead=1    \
  --model='./models/resnet18-race/best.pth'

echo '***********************************************'
echo '***************** Resnet101 *******************'
echo '***********************************************'

echo 'Running multi task learning model'
python3 benchmark.py   \
  --batchSize=1        \
  --inputSize=64       \
  --backbone=resnet101 \
  --ageHead=1          \
  --genderHead=1       \
  --ethnicityHead=1    \
  --model='./models/resnet101/best.pth'

echo 'Running only age'
python3 benchmark.py   \
  --batchSize=1        \
  --inputSize=64       \
  --backbone=resnet101 \
  --ageHead=1          \
  --genderHead=0       \
  --ethnicityHead=0    \
  --model='./models/resnet101-age/best.pth'

echo 'Running gender only'
python3 benchmark.py   \
  --batchSize=1        \
  --inputSize=64       \
  --backbone=resnet101 \
  --ageHead=0          \
  --genderHead=1       \
  --ethnicityHead=0    \
  --model='./models/resnet101-gender/best.pth'

echo 'Running ethnicity only'
python3 benchmark.py   \
  --batchSize=1        \
  --inputSize=64       \
  --backbone=resnet101 \
  --ageHead=0          \
  --genderHead=0       \
  --ethnicityHead=1    \
  --model='./models/resnet101-race/best.pth'