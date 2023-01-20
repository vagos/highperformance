% File       : show_image.m
% Description: Show gray-scale image from binary file
clear('all');

args = argv();
filename = args{1};
out_filename = args{2};

A = load(filename);
imwrite(A / 255, out_filename);
