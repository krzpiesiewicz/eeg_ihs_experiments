#!/bin/bash
for img in *.png;
do
  convert -colorspace gray $img "grayscale/${img}";
done;
