clear; clc;

addpath utils\

x = 1:1000;
y = -x;

window = 10;
delays = [0 5 10]; 

Xemb = EmbeddingDelayed(x', window, delays, 1);
Yemb = EmbeddingDelayed(y', window, delays, 1);

