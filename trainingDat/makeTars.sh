#!/bin/bash
for i in *
do
    tar -czf $i.tar $i
done
