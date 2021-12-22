#!/bin/bash
export CUDA_DEVICE_ODER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

python train_task.py ./output