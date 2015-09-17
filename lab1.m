%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Course: Introduction to Deep Learning
% Teacher: Zhang Yi
% Student:
% ID:
%
% Lab 1 - Getting familiar with MATLAB
%
% Task:
% 1. Read in image files
% 2. Transform an image to an vector
% 3. Reorganize the vectors in a matrix
% 4. Transform digit to label format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc,clear;

% indicate the directory of images
image_list = dir('img/*.jpg');

% data and label matrix to be filled
data = [];

% ��ʼ��
I = imread('img/9-5.jpg');
[h,w,c] = size(I);
img_size = w*h;
mat = zeros(img_size,1);
mat_1 = reshape(mat,h,w);
mat_2 = reshape(mat_1,h*w,1);
p = zeros(h*w,numel(image_list));

% for each image file
for i = 1:numel(image_list)
    image_name = ['img/',image_list(i).name];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code BELOW
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 1. Read in image files (use imread() function)
    I = imread(image_name);
    mat_1 = im2bw(I);
%     imshow(mat_1);
    % 2. Transform an image to an vector
    mat_2 = reshape(mat_1,h*w,1);
    % 3. Reorgnize the vectors in a matrix
    data = [data mat_2];
    %�������ֵ
    switch i
      case {1,2,3}
          t(i)=0;    %����0
      case{4,5,6}
          t(i)=1 ; %����1
      case{7,8,9}
          t(i)=2  ;  %����2
      case{10,11,12}
          t(i)=3   ; %����3
      case{13,14,15}
          t(i)=4  ;  %����4
      case{16,17,18}
          t(i)=5   ; %����5
      case{19,20,21}
          t(i)=6    ;%����6
      case{22,23,24}
          t(i)=7;    %����7
      case{25,26,27}
          t(i)=8 ;   %����8
      case{28,29,30}
          t(i)=9  ;  %����9
    end                    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Your code ABOVE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
% save 51ET data t;
% load  51ET data t;           %��������
pr(1:784,1)=0;
pr(1:784,2)=1;
net=newff(pr,[25 1],{'logsig' 'purelin'},'traingdx','learngdm'); %����BP����
net.trainParam.epochs=3000;   %����ѵ������
net.trainParam.goal=0.000000001;   %����ѵ��Ŀ��
net.trainParam.show=10;       %����ѵ����ʾ����
net.trainParam.lr=0.05        %����ѵ��ѧϰ��
net=train(net,data,t);           %ѵ��BP����
% save ET51net net;

%���Բ��Լ�
image_list = dir('text/*.jpg');
% data and label matrix to be filled
data2 = [];
label = [];
%��ʼ��
I = imread('img/9-5.jpg');
[h,w,c] = size(I);
img_size = w*h;
mat = zeros(img_size,1);
mat_1 = reshape(mat,h,w);
mat_2 = reshape(mat_1,h*w,1);
p = zeros(h*w,numel(image_list));
% for each image file
for i = 1:numel(image_list)
    image_name = ['text/',image_list(i).name];
    % 1. Read in image files (use imread() function)
    I = imread(image_name);
    mat_1 = im2bw(I);   
    % 2. Transform an image to an vector
    mat_2 = reshape(mat_1,h*w,1);
    data2 = [data2 mat_2];
    % ��������
    digit = sim(net,mat_2);        
    digit_label = round(digit);  
    % Reorgnize labels in a matrix
    label = [label digit_label];
end

% Show results
close all
rows = 5; % display in 5 rows
figure(1);
for i = 1:numel(image_list)
    subplot(rows, ceil(numel(image_list)/rows), i);
    imshow(reshape(data2(:, i), 28, []));
    title((label(:, i))); 
end