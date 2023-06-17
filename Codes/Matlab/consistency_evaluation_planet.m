%% consistency evaluation
clc;
clear;

%%
all(:,:,1)=imread('C:\Users\bnikparv\Downloads\plnt\results\nov15_cls.tif');
all(:,:,2)=imread('C:\Users\bnikparv\Downloads\plnt\results\jan16_cls.tif');
all(:,:,3)=imread('C:\Users\bnikparv\Downloads\plnt\results\feb16_cls.tif');
all(:,:,4)=imread('C:\Users\bnikparv\Downloads\plnt\results\apr16_cls.tif');
all(:,:,5)=imread('C:\Users\bnikparv\Downloads\plnt\results\jun16_cls.tif');
all(:,:,6)=imread('C:\Users\bnikparv\Downloads\plnt\results\jul16_cls.tif');
all(:,:,7)=imread('C:\Users\bnikparv\Downloads\plnt\results\sep16_cls.tif');
all(:,:,8)=imread('C:\Users\bnikparv\Downloads\plnt\results\oct16_cls.tif');
all(:,:,9)=imread('C:\Users\bnikparv\Downloads\plnt\results\nov16_cls.tif');
all(:,:,10)=imread('C:\Users\bnikparv\Downloads\plnt\results\dec16_cls.tif');
all(:,:,11)=imread('C:\Users\bnikparv\Downloads\plnt\results\feb17_cls.tif');
all(:,:,12)=imread('C:\Users\bnikparv\Downloads\plnt\results\mar17_cls.tif');
all(:,:,13)=imread('C:\Users\bnikparv\Downloads\plnt\results\apr17_cls.tif');
all(:,:,14)=imread('C:\Users\bnikparv\Downloads\plnt\results\may17_cls.tif');
all(:,:,15)=imread('C:\Users\bnikparv\Downloads\plnt\results\jun17_cls.tif');
all(:,:,16)=imread('C:\Users\bnikparv\Downloads\plnt\results\jul17_cls.tif');
all(:,:,17)=imread('C:\Users\bnikparv\Downloads\plnt\results\aug17_cls.tif');
all(:,:,18)=imread('C:\Users\bnikparv\Downloads\plnt\results\sep17_cls.tif');
all(:,:,19)=imread('C:\Users\bnikparv\Downloads\plnt\results\oct17_cls.tif');
all(:,:,20)=imread('C:\Users\bnikparv\Downloads\plnt\results\nov17_cls.tif');
all(:,:,21)=imread('C:\Users\bnikparv\Downloads\plnt\results\dec17_cls.tif');
all(:,:,22)=imread('C:\Users\bnikparv\Downloads\plnt\results\jan18_cls.tif');
all(:,:,23)=imread('C:\Users\bnikparv\Downloads\plnt\results\feb18_cls.tif');
all(:,:,24)=imread('C:\Users\bnikparv\Downloads\plnt\results\mar18_cls.tif');
all(:,:,25)=imread('C:\Users\bnikparv\Downloads\plnt\results\apr18_cls.tif');
all(:,:,26)=imread('C:\Users\bnikparv\Downloads\plnt\results\may18_cls.tif');
all(:,:,27)=imread('C:\Users\bnikparv\Downloads\plnt\results\jun18_cls.tif');
all(:,:,28)=imread('C:\Users\bnikparv\Downloads\plnt\results\jul18_cls.tif');
all(:,:,29)=imread('C:\Users\bnikparv\Downloads\plnt\results\aug18_cls.tif');
all(:,:,30)=imread('C:\Users\bnikparv\Downloads\plnt\results\sep18_cls.tif');
all(:,:,31)=imread('C:\Users\bnikparv\Downloads\plnt\results\oct18_cls.tif');
all(:,:,32)=imread('C:\Users\bnikparv\Downloads\plnt\results\nov18_cls.tif');
all(:,:,33)=imread('C:\Users\bnikparv\Downloads\plnt\results\dec18_cls.tif');

all = uint8(all);

trajectories = reshape(all, [4242*4951, 33]);
trajchange(:,1) = trajectories(:,2)-trajectories(:,1);
trajchange(:,2) = trajectories(:,3)-trajectories(:,2);
trajchange(:,3) = trajectories(:,4)-trajectories(:,3);
trajchange(:,4) = trajectories(:,5)-trajectories(:,4);
trajchange(:,5) = trajectories(:,6)-trajectories(:,5);
trajchange(:,6) = trajectories(:,7)-trajectories(:,6);
trajchange(:,7) = trajectories(:,8)-trajectories(:,7);
trajchange(:,8) = trajectories(:,9)-trajectories(:,8);
trajchange(:,9) = trajectories(:,10)-trajectories(:,9);
trajchange(:,10) = trajectories(:,11)-trajectories(:,10);
trajchange(:,11) = trajectories(:,12)-trajectories(:,11);
trajchange(:,12) = trajectories(:,13)-trajectories(:,12);
trajchange(:,13) = trajectories(:,14)-trajectories(:,13);
trajchange(:,14) = trajectories(:,15)-trajectories(:,14);
trajchange(:,15) = trajectories(:,16)-trajectories(:,15);
trajchange(:,16) = trajectories(:,17)-trajectories(:,16);
trajchange(:,17) = trajectories(:,18)-trajectories(:,17);
trajchange(:,18) = trajectories(:,19)-trajectories(:,18);
trajchange(:,19) = trajectories(:,20)-trajectories(:,19);
trajchange(:,20) = trajectories(:,21)-trajectories(:,20);
trajchange(:,21) = trajectories(:,22)-trajectories(:,21);
trajchange(:,22) = trajectories(:,23)-trajectories(:,22);
trajchange(:,23) = trajectories(:,24)-trajectories(:,23);
trajchange(:,24) = trajectories(:,25)-trajectories(:,24);
trajchange(:,25) = trajectories(:,26)-trajectories(:,25);
trajchange(:,26) = trajectories(:,27)-trajectories(:,26);
trajchange(:,27) = trajectories(:,28)-trajectories(:,27);
trajchange(:,28) = trajectories(:,29)-trajectories(:,28);
trajchange(:,29) = trajectories(:,30)-trajectories(:,29);
trajchange(:,30) = trajectories(:,31)-trajectories(:,30);
trajchange(:,31) = trajectories(:,32)-trajectories(:,31);
trajchange(:,32) = trajectories(:,33)-trajectories(:,32);

trajchange(trajchange~=0)=1;

chng = sum(trajchange, 2);
for i=1:size(trajectories, 1)
   unqclssnum(i, 1) = size(unique(trajectories(i,:)), 2);
end

for i=1:size(trajectories, 1)
   aux = unique(trajectories(i,:));
   j=1;
   ss = '';
   while j<=size(aux, 2)
       ss = ss + string(aux(1,j));
       j=j+1;
   end
   unqclss(i, 1) = ss;
end

save('G:\backupC27152020\C\models2\planetImage\classificationcode\saves\variables1.mat','all',...
    'trajectories','trajchange','chng','unqclssnum','unqclss');

%% 
clc;
clear;
load('G:\backupC27152020\C\models2\planetImage\classificationcode\saves\variables1.mat','all',...
    'trajectories','trajchange','chng','unqclssnum','unqclss');

modes=zeros(size(trajectories));
for i=1:size(trajectories, 1)
   if sum(trajchange(i,:))>=0
       i
       for j=1:size(trajectories, 2)
           switch j
           case 1
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./idx_bg));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./idx_mx));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./idx_rh));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./idx_rn));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 2
              dd = [1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25 26 27 28 29 30 31];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 3
              dd = [2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25 26 27 28 29 30];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 4
              dd = [3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25 26 27 28 29];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 5
              dd = [4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25 26 27 28];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 6
              dd = [5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25 26 27];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 7
              dd = [6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25 26];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 8
              dd = [7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24 25];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 9
              dd = [8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23 24];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 10
              dd = [9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22 23];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 11
              dd = [10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21 22];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 12
              dd = [11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  21];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 13
              dd = [12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ...
                  ];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 14
              dd = [13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 15
              dd = [14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 16
              dd = [15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 17
              dd = [16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 18
              dd = [17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 19
              dd = [18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 20
              dd = [19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 21
              dd = [20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11 12];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 22
              dd = [21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10 11];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx; 
           case 23
              dd = [22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9 10];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 24
              dd = [23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8 9];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 25
              dd = [24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7 8];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 26
              dd = [25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6 7];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 27
              dd = [26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5 6];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 28
              dd = [27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4 5];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 29
              dd = [28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3 4];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 30
              dd = [29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2 3];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 31
              dd = [30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1 2];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           case 32
              dd = [31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0 1];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;            
           otherwise
              dd = [32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0];
              idx_bg = find(trajectories(i,:)==1);
              bg = (sum(1./(dd(idx_bg)+1)));
              idx_mx = find(trajectories(i,:)==2);
              mx = (sum(1./(dd(idx_mx)+1)));
              idx_rh = find(trajectories(i,:)==3);
              rh = (sum(1./(dd(idx_rh)+1)));
              idx_rn = find(trajectories(i,:)==4);
              rn = (sum(1./(dd(idx_rn)+1)));
              [~, idx] = max([bg mx rh rn]);
              %modes(i,j,:)=[1 bg 2 mx 3 rh 4 rn];
              modes(i,j)=idx;
           end
       end
   else
       modes(i,:)=trajectories(i,:);
   end
end

save('G:\backupC27152020\C\models2\planetImage\classificationcode\saves\variables2.mat','modes', '-v7.3');
%%
modes2=reshape(modes, [4242 4951 33]);

for i=1:size(modes2, 3)
    a = uint8(modes2(:,:,i));
    name = 'C:\Users\bnikparv\Downloads\plnt\results\corrections\temporal\time_' + string(i) + '.txt'
    fid = fopen(name, 'wt');
    % print a title, followed by a blank line
    % 
    fprintf(fid, 'NCOLS 4951\nNROWS 4242\nXLLCORNER 325329\nYLLCORNER 4018638\n');
    fprintf(fid, 'CELLSIZE 3\nNODATA_VALUE 0\n');
    % fprintf(fid, 'NCOLS 29630\nNROWS 25353\nXLLCORNER 325355.95\nYLLCORNER 4031346.05\n');
    % fprintf(fid, 'CELLSIZE 0.499991422727483\nNODATA_VALUE 0\n');
    for ii = 1:size(a,1)
        fprintf(fid,'%g ',a(ii,:));
        fprintf(fid,'\n');
    end
    fclose(fid);
end