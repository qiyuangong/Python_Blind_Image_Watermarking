%% 傅里叶变换加水印源代码
 
%% 运行环境 Matlab2010a
 
clc;clear;close all;
alpha = 1;
%% read data
 
im = double(imread('gl1.jpg'))/255;
mark = double(imread('watermark.jpg'))/255;
figure, imshow(im),title('original image');
figure, imshow(mark),title('watermark');
%% encode mark
 
imsize = size(im);
%random
 
TH=zeros(imsize(1)*0.5,imsize(2),imsize(3));
TH1 = TH;
TH1(1:size(mark,1),1:size(mark,2),:) = mark;
M=randperm(0.5*imsize(1));
N=randperm(imsize(2));
save('encode.mat','M','N');
for i=1:imsize(1)*0.5
for j=1:imsize(2)
TH(i,j,:)=TH1(M(i),N(j),:);
end
end
% symmetric
 
mark_ = zeros(imsize(1),imsize(2),imsize(3));
mark_(1:imsize(1)*0.5,1:imsize(2),:)=TH;
for i=1:imsize(1)*0.5
for j=1:imsize(2)
mark_(imsize(1)+1-i,imsize(2)+1-j,:)=TH(i,j,:);
end
end
figure,imshow(mark_),title('encoded watermark');
%imwrite(mark_,'encoded watermark.jpg');
 
%% add watermark
 
FA=fft2(im);
figure,imshow(FA);title('spectrum of original image');
FB=FA+alpha*double(mark_);
figure,imshow(FB); title('spectrum of watermarked image');
FAO=ifft2(FB);
figure,imshow(FAO); title('watermarked image');
%imwrite(uint8(FAO),'watermarked image.jpg');
 
RI = FAO-double(im);
figure,imshow(uint8(RI)); title('residual');
%imwrite(uint8(RI),'residual.jpg');
 
xl = 1:imsize(2);
yl = 1:imsize(1);
[xx,yy] = meshgrid(xl,yl);
figure, plot3(xx,yy,FA(:,:,1).^2+FA(:,:,2).^2+FA(:,:,3).^2),title('spectrum of original image');
figure, plot3(xx,yy,FB(:,:,1).^2+FB(:,:,2).^2+FB(:,:,3).^2),title('spectrum of watermarked image');
figure, plot3(xx,yy,FB(:,:,1).^2+FB(:,:,2).^2+FB(:,:,3).^2-FA(:,:,1).^2+FA(:,:,2).^2+FA(:,:,3).^2),title('spectrum of watermark');
%% extract watermark
 
FA2=fft2(FAO);
G=(FA2-FA)/alpha;
GG=G;
for i=1:imsize(1)*0.5
for j=1:imsize(2)
GG(M(i),N(j),:)=G(i,j,:);
end
end
for i=1:imsize(1)*0.5
for j=1:imsize(2)
GG(imsize(1)+1-i,imsize(2)+1-j,:)=GG(i,j,:);
end
end
figure,imshow(GG);title('extracted watermark');
%imwrite(uint8(GG),'extracted watermark.jpg');
 
%% MSE and PSNR
 
C=double(im);
RC=double(FAO);
MSE=0; PSNR=0;
for i=1:imsize(1)
for j=1:imsize(2)
MSE=MSE+(C(i,j)-RC(i,j)).^2;
end
end
MSE=MSE/360.^2;
PSNR=20*log10(255/sqrt(MSE));
MSE
PSNR
 
%% attack test
 
%% attack by smearing
 
%A = double(imread('gl1.jpg'));
 
%B = double(imread('attacked image.jpg'));
 
attack = 1-double(imread('attack.jpg'))/255;
figure,imshow(attack);
FAO_ = FAO;
for i=1:imsize(1)
for j=1:imsize(2)
if attack(i,j,1)+attack(i,j,2)+attack(i,j,3)>0.5
FAO_(i,j,:) = attack(i,j,:);
end
end
end
figure,imshow(FAO_);
%extract watermark
 
FA2=fft2(FAO_);
G=(FA2-FA)*2;
GG=G;
for i=1:imsize(1)*0.5
for j=1:imsize(2)
GG(M(i),N(j),:)=G(i,j,:);
end
end
for i=1:imsize(1)*0.5
for j=1:imsize(2)
GG(imsize(1)+1-i,imsize(2)+1-j,:)=GG(i,j,:);
end
end
figure,imshow(GG);title('extracted watermark');
 
%% attack by cutting
 
s2 = 0.8;
FAO_ = FAO;
FAO_(:,s2*imsize(2)+1:imsize(2),:) = FAO_(:,1:int32((1-s2)*imsize(2)),:);
figure,imshow(FAO_);
%extract watermark
 
FA2=fft2(FAO_);
G=(FA2-FA)*2;
GG=G;
for i=1:imsize(1)*0.5
for j=1:imsize(2)
GG(M(i),N(j),:)=G(i,j,:);
end
end
for i=1:imsize(1)*0.5
for j=1:imsize(2)
GG(imsize(1)+1-i,imsize(2)+1-j,:)=GG(i,j,:);
end
end
figure,imshow(GG);title('extracted watermark');
 
%%小波变换加水印，解水印大家按照加的思路逆过来就好
 
clc;clear;close all;
%% read data
 
im = double(imread('gl1.jpg'))/255;
mark = double(imread('watermark.jpg'))/255;
figure, imshow(im),title('original image');
figure, imshow(mark),title('watermark');
%% RGB division
 
im=double(im);
mark=double(mark);
imr=im(:,:,1);
markr=mark(:,:,1);
img=im(:,:,2);
markg=mark(:,:,2);
imb=im(:,:,3);
markb=mark(:,:,3);
%% parameter
 
r=0.04;
g = 0.04;
b = 0.04;
%% wavelet tranform and add watermark
 
% for red
 
[Cwr,Swr]=wavedec2(markr,1,'haar');
[Cr,Sr]=wavedec2(imr,2,'haar');
% add watermark
 
Cr(1:size(Cwr,2)/16)=...
Cr(1:size(Cwr,2)/16)+r*Cwr(1:size(Cwr,2)/16);
k=0;
while k<=size(Cr,2)/size(Cwr,2)-1
Cr(1+size(Cr,2)/4+k*size(Cwr,2)/4:size(Cr,2)/4+...
(k+1)*size(Cwr,2)/4)=Cr(1+size(Cr,2)/4+...
k*size(Cwr,2)/4:size(Cr,2)/4+(k+1)*size(Cwr,2)/4)+...
r*Cwr(1+size(Cwr,2)/4:size(Cwr,2)/2);
Cr(1+size(Cr,2)/2+k*size(Cwr,2)/4:size(Cr,2)/2+...
(k+1)*size(Cwr,2)/4)=Cr(1+size(Cr,2)/2+...
k*size(Cwr,2)/4:size(Cr,2)/2+(k+1)*size(Cwr,2)/4)+...
r*Cwr(1+size(Cwr,2)/2:3*size(Cwr,2)/4);
Cr(1+3*size(Cwr,2)/4+k*size(Cwr,2)/4:3*size(Cwr,2)/4+...
(k+1)*size(Cwr,2)/4)=Cr(1+3*size(Cr,2)/4+...
k*size(Cwr,2)/4:3*size(Cr,2)/4+(k+1)*size(Cwr,2)/4)+...
r*Cwr(1+3*size(Cwr,2)/4:size(Cwr,2));
k=k+1;
end;
Cr(1:size(Cwr,2)/4)=Cr(1:size(Cwr,2)/4)+r*Cwr(1:size(Cwr,2)/4);
 
% for green
 
[Cwg,Swg]=WAVEDEC2(markg,1,'haar');
[Cg,Sg]=WAVEDEC2(img,2,'haar');
Cg(1:size(Cwg,2)/16)=...
Cg(1:size(Cwg,2)/16)+g*Cwg(1:size(Cwg,2)/16);
k=0;
while k<=size(Cg,2)/size(Cwg,2)-1
Cg(1+size(Cg,2)/4+k*size(Cwg,2)/4:size(Cg,2)/4+...
(k+1)*size(Cwg,2)/4)=Cg(1+size(Cg,2)/4+...
k*size(Cwg,2)/4:size(Cg,2)/4+(k+1)*size(Cwg,2)/4)+...
g*Cwg(1+size(Cwg,2)/4:size(Cwg,2)/2);
Cg(1+size(Cg,2)/2+k*size(Cwg,2)/4:size(Cg,2)/2+...
(k+1)*size(Cwg,2)/4)=Cg(1+size(Cg,2)/2+...
k*size(Cwg,2)/4:size(Cg,2)/2+(k+1)*size(Cwg,2)/4)+...
g*Cwg(1+size(Cwg,2)/2:3*size(Cwg,2)/4);
Cg(1+3*size(Cg,2)/4+k*size(Cwg,2)/4:3*size(Cg,2)/4+...
(k+1)*size(Cwg,2)/4)=Cg(1+3*size(Cg,2)/4+...
k*size(Cwg,2)/4:3*size(Cg,2)/4+(k+1)*size(Cwg,2)/4)+...
g*Cwg(1+3*size(Cwg,2)/4:size(Cwg,2));
k=k+1;
end;
Cg(1:size(Cwg,2)/4)=Cg(1:size(Cwg,2)/4)+g*Cwg(1:size(Cwg,2)/4);
 
% for blue
 
[Cwb,Swb]=WAVEDEC2(markb,1,'haar');
[Cb,Sb]=WAVEDEC2(imb,2,'haar');
Cb(1:size(Cwb,2)/16)+b*Cwb(1:size(Cwb,2)/16);
k=0;
while k<=size(Cb,2)/size(Cwb,2)-1
Cb(1+size(Cb,2)/4+k*size(Cwb,2)/4:size(Cb,2)/4+...
(k+1)*size(Cwb,2)/4)=Cb(1+size(Cb,2)/4+...
k*size(Cwb,2)/4:size(Cb,2)/4+(k+1)*size(Cwb,2)/4)+...
g*Cwb(1+size(Cwb,2)/4:size(Cwb,2)/2);
Cb(1+size(Cb,2)/2+k*size(Cwb,2)/4:size(Cb,2)/2+...
(k+1)*size(Cwb,2)/4)=Cb(1+size(Cb,2)/2+...
k*size(Cwb,2)/4:size(Cb,2)/2+(k+1)*size(Cwb,2)/4)+...
b*Cwb(1+size(Cwb,2)/2:3*size(Cwb,2)/4);
Cb(1+3*size(Cb,2)/4+k*size(Cwb,2)/4:3*size(Cb,2)/4+...
(k+1)*size(Cwb,2)/4)=Cb(1+3*size(Cb,2)/4+...
k*size(Cwb,2)/4:3*size(Cb,2)/4+(k+1)*size(Cwb,2)/4)+...
b*Cwb(1+3*size(Cwb,2)/4:size(Cwb,2));
k=k+1;
end;
Cb(1:size(Cwb,2)/4)=Cb(1:size(Cwb,2)/4)+b*Cwb(1:size(Cwb,2)/4);
%% image reconstruction
 
imr=WAVEREC2(Cr,Sr,'haar');
img=WAVEREC2(Cg,Sg,'haar');
imb=WAVEREC2(Cb,Sb,'haar');
imsize=size(imr);
FAO=zeros(imsize(1),imsize(2),3);
for i=1:imsize(1);
for j=1:imsize(2);
FAO(i,j,1)=imr(i,j);
FAO(i,j,2)=img(i,j);
FAO(i,j,3)=imb(i,j);
end
end
figure, imshow(FAO); title('watermarked image');
